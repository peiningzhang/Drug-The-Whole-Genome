#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import pickle
import torch
from unicore import checkpoint_utils, distributed_utils, options, utils
from unicore.logging import progress_bar
from unicore import tasks
import numpy as np
from tqdm import tqdm
import unicore

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.inference")


#from skchem.metrics import bedroc_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment
from sklearn.metrics import roc_curve


import os
from Bio.PDB import PDBParser,Chain,Model,Structure
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB import is_aa
from Bio.PDB.Residue import DisorderedResidue,Residue
from Bio.PDB.Atom import DisorderedAtom
import warnings
from Bio.PDB.StructureBuilder import PDBConstructionWarning
from tqdm import tqdm
import numpy as np
import lmdb
import numpy as np
import pickle
import re

def write_lmdb(data, lmdb_path, num):
    #resume
    
    env = lmdb.open(lmdb_path, subdir=False, readonly=False, lock=False, readahead=False, meminit=False, map_size=1099511627776)
    with env.begin(write=True) as txn:
        for d in data:
            txn.put(str(num).encode('ascii'), pickle.dumps(d))
            num += 1
    
    return num

warnings.filterwarnings(
    action='ignore',
    category=PDBConstructionWarning)
def extract_lig_recpt(biopy_model,ligname):
    lig_list = []
    tmp_chain = Chain.Chain('A') 
    rid = 0
    for chain in biopy_model:
        for res in chain:
            res.detach_parent()
            if not(is_aa(res,standard=True)):
                if res.resname == ligname:
                    lig_list.append((chain.id+'_'+str(res.id[1]),res))
                continue
            if res.is_disordered():
                if isinstance(res,DisorderedResidue):
                    res = res.selected_child
                    res.id = (res.id[0],rid,res.id[2])
                    rid += 1 
                    tmp_chain.add(res.copy())      
                else:
                    new_res = Residue(res.id,res.resname,res.segid)
                    for atom in res:
                        if isinstance(atom,DisorderedAtom):  
                            atom.selected_child.disordered_flag = 0
                            new_res.add(atom.selected_child.copy())
                        else:
                            new_res.add(atom)
                    res = new_res
                    res.id = (res.id[0],rid,res.id[2])
                    rid += 1 
                    tmp_chain.add(res.copy())       
            else:
                res.id = (res.id[0],rid,res.id[2])
                rid += 1 
                tmp_chain.add(res.copy())
    # tmp_structure = Structure.Structure('tmp')
    # tmp_model = Model.Model(0)
    # tmp_structure.add(tmp_model) 
    # tmp_model.add(tmp_chain)
  
    return tmp_chain,lig_list

def get_binding_pockets(biopy_chain,liglist):
    pockets = []
    for n,lig in liglist:
            lig_coord = np.array([i.coord for i in lig.get_atoms() if i.element!='H'])
            tmp_chain = Chain.Chain('A') 
            for res in biopy_chain:
                res_coord = np.array([i.get_coord() for i in res.get_atoms()])
                dist = np.linalg.norm(res_coord[:,None,:]-lig_coord[None,:,:],axis=-1).min()
                if dist<=6:
                    tmp_chain.add(res.copy())
            pockets.append((n,tmp_chain))
    return pockets  

def pocket2lmdb(pocket_name,biopy_chain,pdb_name):
    recpt = list(biopy_chain.get_atoms())
    pocket_atom_type = [x.element for x in recpt if x.element!='H']
    pocket_coord = [x.coord for x in recpt if x.element!='H']
    print(pdb_name+'_'+pocket_name,len(pocket_atom_type))
    return {
        'pocket': pdb_name+'_'+pocket_name,
        'pocket_atoms': pocket_atom_type,
        'pocket_coordinates': pocket_coord
    }


def process_one_pdbdir(dirs,name='pocket'):
    all_pocket = []
    for d in os.listdir(dirs):
        if '.pdb' in d and '_clean' not in d:
            try:
                p = PDBParser()
                model = p.get_structure('0',os.path.join(dirs,d))[0]
                # Try to extract ligand name from filename first
                ligand_match = re.search(r'_.*\.',d)
                if ligand_match:
                    ligand_name = ligand_match[0][1:-1]
                else:
                    # If no match, try to extract from PDB file
                    ligand_name = None
                    with open(os.path.join(dirs,d), 'r') as f:
                        for line in f:
                            if line.startswith('HET '):
                                parts = line.split()
                                if len(parts) >= 2:
                                    ligand_name = parts[1]
                                    break
                            elif line.startswith('HETATM'):
                                parts = line.split()
                                if len(parts) >= 4:
                                    resname = parts[3]
                                    if resname not in ['HOH', 'WAT', 'SO4', 'PO4', 'CL', 'NA']:  # Skip common ions/water
                                        ligand_name = resname
                                        break
                
                if ligand_name:
                    tmp_chain,liglist = extract_lig_recpt(model,ligand_name)
                    pocket = get_binding_pockets(tmp_chain,liglist)
                    pocket = [pocket2lmdb(n,p,d.split('.')[0]) for n,p in pocket]
                    all_pocket += pocket
                else:
                    logger.warning(f"Could not extract ligand name from {d}, skipping")
            except Exception as e:
                logger.warning(f"Error processing {d}: {e}")
                pass
    if os.path.exists(os.path.join(dirs,f'{name}.lmdb')):
        return 0
    if len(all_pocket) == 0:
        error_msg = f"No pockets found in {dirs}. Please check PDB files and ligand names."
        logger.error(error_msg)
        raise ValueError(error_msg)
    write_lmdb(all_pocket,os.path.join(dirs,f'{name}.lmdb'),0)
    return 1



def main(args):

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)


    # Load model
    logger.info("loading model(s) from {}".format(args.path))
    #state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    #model.load_state_dict(state["model"], strict=False)

    # Move models to GPU
    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    # Print args
    logger.info(args)


    model.eval()

    pocket_lmdb_path = os.path.join(args.pocket_dir, "pocket.lmdb")
    if not os.path.exists(pocket_lmdb_path):
        try:
            ret = process_one_pdbdir(args.pocket_dir)
            if not os.path.exists(pocket_lmdb_path):
                raise ValueError(f"Failed to create pocket.lmdb in {args.pocket_dir}. No pockets were extracted from PDB files.")
        except ValueError as e:
            # Re-raise ValueError from process_one_pdbdir
            raise
        except Exception as e:
            raise ValueError(f"Error processing PDB files in {args.pocket_dir}: {e}")
    else:
        # Check if pocket.lmdb is empty or corrupted
        try:
            import lmdb
            env = lmdb.open(pocket_lmdb_path, subdir=False, readonly=True)
            txn = env.begin()
            keys = list(txn.cursor().iternext(values=False))
            env.close()
            if len(keys) == 0:
                logger.warning(f"pocket.lmdb is empty, regenerating...")
                os.remove(pocket_lmdb_path)
                try:
                    ret = process_one_pdbdir(args.pocket_dir)
                    if not os.path.exists(pocket_lmdb_path):
                        raise ValueError(f"Failed to regenerate pocket.lmdb in {args.pocket_dir}. No pockets were extracted from PDB files.")
                except ValueError as e:
                    raise
                except Exception as e:
                    raise ValueError(f"Error regenerating pocket.lmdb: {e}")
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            logger.warning(f"Error checking pocket.lmdb: {e}, regenerating...")
            if os.path.exists(pocket_lmdb_path):
                os.remove(pocket_lmdb_path)
            try:
                ret = process_one_pdbdir(args.pocket_dir)
                if not os.path.exists(pocket_lmdb_path):
                    raise ValueError(f"Failed to regenerate pocket.lmdb in {args.pocket_dir}. No pockets were extracted from PDB files.")
            except ValueError as e:
                raise
            except Exception as e:
                raise ValueError(f"Error regenerating pocket.lmdb: {e}")

    # Verify pocket.lmdb exists and is not empty before proceeding
    if not os.path.exists(pocket_lmdb_path):
        raise ValueError(f"pocket.lmdb does not exist in {args.pocket_dir}. No pockets were extracted from PDB files.")
    
    try:
        import lmdb
        env = lmdb.open(pocket_lmdb_path, subdir=False, readonly=True)
        txn = env.begin()
        keys = list(txn.cursor().iternext(values=False))
        env.close()
        if len(keys) == 0:
            raise ValueError(f"pocket.lmdb is empty. No pockets were extracted from PDB files in {args.pocket_dir}")
    except ImportError:
        pass  # lmdb not available, skip check
    except FileNotFoundError:
        raise ValueError(f"pocket.lmdb does not exist in {args.pocket_dir}. No pockets were extracted from PDB files.")
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Error reading pocket.lmdb: {e}")

    # read pocket dir 
    pocket_reps, pocket_names = task.encode_pockets_multi_folds(model, args.pocket_dir, pocket_lmdb_path)

    # print shape

    logger.info(f"pocket_reps shape: {pocket_reps.shape}")

    # print name

    logger.info(f"pocket_names shape: {pocket_names}")


    # save names and reps together to pickle file

    
    with open(os.path.join(args.pocket_dir, "pocket_reps.pkl"), "wb") as f:
        pickle.dump((pocket_names, pocket_reps), f)
   

    


def cli_main():
    # add args
    

    parser = options.get_validation_parser()
    parser.add_argument("--pocket-dir", type=str, default="", help="path for pocket dir")

    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()

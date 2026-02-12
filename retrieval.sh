#!/bin/bash
# If you set use_cache=True, then we will use the pre-encoded mols for screening.
# This is default for all the wet-lab experiment targets.
# Else, please set the MOL_PATH to a lmdb path as the screening library.

echo "First argument: $1"

MOL_PATH="mols.lmdb" # path to the molecule file
POCKET_PATH="./data/targets/NET/pocket.lmdb"  # Correct path format: ./data/targets/{target}/pocket.lmdb
FOLD_VERSION=6_folds
use_cache=True
save_path="NET.txt"




CUDA_VISIBLE_DEVICES="0" python ./unimol/retrieval.py --user-dir ./unimol --valid-subset test \
       --num-workers 8 --ddp-backend=c10d --batch-size 4 \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 511 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --log-interval 100 --log-format simple \
       --mol-path $MOL_PATH \
       --pocket-path $POCKET_PATH \
       --fold-version $FOLD_VERSION \
       --use-cache $use_cache \
       --save-path $save_path \
       "./dict"
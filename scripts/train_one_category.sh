#!/bin/bash

# run training on one category of BBD everyday subset

#######################################################################
# An example usage:
#     ./scripts/train_one_category.sh "GPUS=1 CPUS_PER_GPU=8 MEM_PER_CPU=5 QOS=normal REPEAT=3 ./scripts/dup_run_sbatch.sh rtx6000 global-everyday-xxx-CATEGORY ./scripts/train.py config.py --fp16 --cudnn" config.py Bottle
#######################################################################

CMD=$1
CFG=$2
cat=$3

cfg="${CFG:0:(-3)}-$cat.py"
cp $CFG $cfg
cmd="${CMD/$CFG/$cfg}"
cmd="${cmd/CATEGORY/$cat}"
cmd="$cmd --category $cat"
eval $cmd

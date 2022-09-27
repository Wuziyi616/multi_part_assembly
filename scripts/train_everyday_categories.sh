#!/bin/bash

# automatically run training on all the categories of BBD everyday subset

#######################################################################
# An example usage:
#     ./scripts/train_everyday_categories.sh "GPUS=1 CPUS_PER_GPU=8 MEM_PER_CPU=5 QOS=normal REPEAT=3 ./scripts/dup_run_sbatch.sh rtx6000 global-everyday-xxx-CATEGORY ./scripts/train.py config.py --fp16 --cudnn" config.py
#######################################################################

CMD=$1
CFG=$2

for cat in BeerBottle Bottle Bowl Cookie Cup DrinkBottle DrinkingUtensil Mirror Mug PillBottle Plate Ring Spoon Statue Teacup Teapot ToyFigure Vase WineBottle WineGlass
do
    cfg="${CFG:0:(-3)}-$cat.py"
    cp $CFG $cfg
    cmd="${CMD/$CFG/$cfg}"
    cmd="${cmd/CATEGORY/$cat}"
    cmd="$cmd --category $cat"
    eval $cmd
done

#!/bin/bash

# This is a wrapper for `sbatch_run.sh` to run repeated experiments
# It will duplicate the same params file for several times and run them all

#######################################################################
# An example usage:
#     GPUS=1 CPUS_PER_TASK=8 MEM_PER_CPU=6 QOS=normal REPEAT=5 ./dup_run_sbatch.sh \
#       rtx6000 test-sbatch test.py ddp params.py --fp16
#######################################################################

# read args from command line
GPUS=${GPUS:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
MEM_PER_CPU=${MEM_PER_CPU:-8}
QOS=${QOS:-normal}
REPEAT=${REPEAT:-5}

PY_ARGS=${@:6}
PARTITION=$1
JOB_NAME=$2
PY_FILE=$3
DDP=$4
PARAMS=$5

for repeat_idx in $(seq 1 $REPEAT)
do
    params="dup${repeat_idx}-${PARAMS}"
    cp $PARAMS $params
    job_name="dup${repeat_idx}-${JOB_NAME}"
    echo "./sbatch_run.sh $PARTITION $job_name $PY_FILE $DDP --params $params $PY_ARGS"
    ./sbatch_run.sh $PARTITION $job_name $PY_FILE $DDP --params $params $PY_ARGS
done

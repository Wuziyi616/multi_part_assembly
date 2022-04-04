#!/bin/bash

# SBATCH file can't directly take command args
# as a workaround, I first use a sh script to read in args
# and then create a new .slrm file for SBATCH execution

#######################################################################
# An example usage:
#     GPUS=1 CPUS_PER_TASK=8 MEM_PER_CPU=6 QOS=normal ./script/sbatch_run.sh rtx6000 train-sbatch ./script/train.py --params params.py
#######################################################################

# read args from command line
GPUS=${GPUS:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
MEM_PER_CPU=${MEM_PER_CPU:-8}
QOS=${QOS:-normal}

PY_ARGS=${@:4}
PARTITION=$1
JOB_NAME=$2
PY_FILE=$3

SLRM_NAME="${JOB_NAME/\//"_"}"
LOG_DIR=checkpoint/$JOB_NAME
DATETIME=$(date "+%Y-%m-%d_%H:%M:%S")
LOG_FILE=$LOG_DIR/${DATETIME}.log

# set up log output folder
mkdir -p $LOG_DIR

# write to new file
echo "#!/bin/bash

# set up SBATCH args
#SBATCH --job-name=$SLRM_NAME
#SBATCH --output=$LOG_FILE
#SBATCH --error=$LOG_FILE
#SBATCH --open-mode=append
#SBATCH --partition=$PARTITION                       # self-explanatory, set to your preference (e.g. gpu or cpu on MaRS, p100, t4, or cpu on Vaughan)
#SBATCH --cpus-per-task=$CPUS_PER_TASK               # self-explanatory, set to your preference
#SBATCH --ntasks=$GPUS
#SBATCH --ntasks-per-node=$GPUS
#SBATCH --mem-per-cpu=${MEM_PER_CPU}G                # self-explanatory, set to your preference
#SBATCH --gres=gpu:$GPUS                             # NOTE: you need a GPU for CUDA support; self-explanatory, set to your preference 
#SBATCH --nodes=1
#SBATCH --qos=$QOS                                   # for 'high' and 'deadline' QoS, refer to https://support.vectorinstitute.ai/AboutVaughan2
#SBATCH --exclude=gpu[179,176,165,126,127,180]

# link /checkpoint to current folder
# ln -sfn /checkpoint/\$USER/\$SLURM_JOB_ID $LOG_DIR

# log some necessary environment params
echo \$SLURM_JOB_ID >> $LOG_FILE                      # log the job id
echo \$SLURM_JOB_PARTITION >> $LOG_FILE               # log the job partition

echo $CONDA_PREFIX >> $LOG_FILE                      # log the active conda environment 

python --version >> $LOG_FILE                        # log Python version
gcc --version >> $LOG_FILE                           # log GCC version
nvcc --version >> $LOG_FILE                          # log NVCC version

# run python file
python $PY_FILE $PY_ARGS >> $LOG_FILE                # the script above, with its standard output appended log file

" >> ./run-${SLRM_NAME}.slrm

# run the created file
sbatch run-${SLRM_NAME}.slrm

# delete it
sleep 1
rm -f run-${SLRM_NAME}.slrm

#!/bin/bash
#SBATCH -p gpu-v100
#SBATCH --ntasks=1        # Number of tasks (see below)
#SBATCH --cpus-per-task=8     # Number of CPU cores per task
#SBATCH --nodes=1         # Ensure that all cores are on one machine
#SBATCH --time=3-00:00      # Runtime in D-HH:MM
#SBATCH --gres=gpu:1      # optionally type and number of gpus
#SBATCH --mem=32G         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=slurm_out/logs_%a_%A.out# File to which STDOUT will be written
#SBATCH --error=slurm_out/errors_%a_%A.err # File to which STDERR will be written
#SBATCH --array=0-1   # maps 0 to 15 to SLURM_ARRAY_TASK_ID below
# print info about current job
scontrol show job $SLURM_JOB_ID


MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
TRAIN_FLAGS="--lr 1e-5 --batch_size 4"
DATA="/mnt/qb/eyepacs/"
IMAGES_ID_FILE=("/mnt/qb/eyepacs/data_processed/metadata/metadata_image_circular_crop.csv"
		"/mnt/qb/eyepacs/data_processed/metadata/splits_circular_crop/train.csv")
STORE_DIR=("/mnt/qb/berens/users/iilanchezian63/diffusion_model_logs/no_rotate_5_class_balancing/"
	   "/mnt/qb/berens/users/iilanchezian63/diffusion_model_logs/no_rotate_5_class_balancing_only_train/")
NUM_GPUS=1

module load mpi/openmpi3-x86_64
sudo /opt/eyepacs/start_eyepacs_mount.sh

#echo ${IMAGES_ID_FILE[$SLURM_ARRAY_TASK_ID]}
mpiexec -n $NUM_GPUS python scripts/image_train.py --images_id_file ${IMAGES_ID_FILE[$SLURM_ARRAY_TASK_ID]} --storage_dir ${STORE_DIR[$SLURM_ARRAY_TASK_ID]} $MODEL_FLAGS $TRAIN_FLAGS

sudo /opt/eyepacs/stop_eyepacs_mount.sh

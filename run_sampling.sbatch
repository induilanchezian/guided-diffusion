#!/bin/bash
#SBATCH -p gpu-v100
#SBATCH --ntasks=1        # Number of tasks (see below)
#SBATCH --cpus-per-task=8     # Number of CPU cores per task
#SBATCH --nodes=1         # Ensure that all cores are on one machine
#SBATCH --time=3-00:00      # Runtime in D-HH:MM
#SBATCH --gres=gpu:1      # optionally type and number of gpus
#SBATCH --mem=32G         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=slurm_out/logs.out# File to which STDOUT will be written
#SBATCH --error=slurm_out/errors.err # File to which STDERR will be written
# print info about current job
scontrol show job $SLURM_JOB_ID


MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--batch_size 4"
DATA="/mnt/qb/eyepacs/"
IMAGES_ID_FILE="/mnt/qb/eyepacs/data_processed/metadata/metadata_image_circular_crop.csv"
MODEL_PATH="/mnt/qb/berens/users/iilanchezian63/diffusion_model_logs/try1/ema_0.9999_300000.pt"
SAVEPATH="/mnt/qb/berens/users/iilanchezian63/diffusion_model_samples/"
NUM_GPUS=1

module load mpi/openmpi3-x86_64
sudo /opt/eyepacs/start_eyepacs_mount.sh

mpiexec -n $NUM_GPUS python scripts/image_sample.py --model_path $MODEL_PATH --num_samples 1000 $MODEL_FLAGS $DIFFUSION_FLAGS --savepath $SAVEPATH --timestep_respacing 500

sudo /opt/eyepacs/stop_eyepacs_mount.sh

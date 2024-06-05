#!/bin/bash
#SBATCH -o ./output/job.%j.out
#SBATCH -p compute1
#SBATCH --qos=normal
#SBATCH -J FirstJob
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 
#SBATCH --mail-type=all
#SBATCH --mail-user=gzben01@gmail.com
#SBATCH -w node8
#SBATCH --mem-per-cpu 10000

python -u scripts/train.py \
    --root .. \
    --no-pretrains \
    --checkpoint_file ../model_erm_r18_10200.pth \
    --chk_frq 100
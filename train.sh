#!/bin/bash
#SBATCH --job-name=yolo_trash
#SBATCH --output=logs/yolo_trash_%j.out
#SBATCH --error=logs/yolo_trash_%j.err
#SBATCH --time=6:00:00
#SBATCH -A eecs
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Uncomment this section if running this as an asyncronous sbatch script -------------------------------------------------------------------
# module load cuda/12.8
# source (your_path_to_conda.sh)/conda.sh
# conda activate (your env name)
#-------------------------------------------------------------------------------------------------------------------------------------------------------

group="runs_150_1"
run_name="sgd_haze"
# "SGD" or "AdamW" (there are others but I haven't used them)
optimizer="SGD"
yolo_augmentations=(
    "translate=0.1"
    "scale=0.3"
    "fliplr=0.5"
    "degrees=10"
    "hsv_h=0.015" 
    "hsv_s=0.4" 
    "hsv_v=0.4"
)

python main.py \
    --yolo_augmentations "${yolo_augmentations[@]}" \
    --epochs 150 \
    --wandb_group "$group" \
    --wandb_name "$run_name" \
    --optimizer "$optimizer" \
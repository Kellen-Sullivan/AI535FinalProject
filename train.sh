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
# source /nfs/hpc/share/gabrieai/miniconda3/etc/profile.d/conda.sh
# conda activate ai_535
#-------------------------------------------------------------------------------------------------------------------------------------------------------

model_name="yolo11l-seg.pt"
group="dif_model_200_runs"
run_name="large"
# "SGD" or "AdamW" (there are others but I haven't used them)
optimizer="SGD"

augmentations=(
    # "color_attenuation"
    # "haze"
)

python main.py \
    --model_name "$model_name" \
    --augmentations "${augmentations[@]}" \
    --epochs 200 \
    --wandb_group "$group" \
    --wandb_name "$run_name" \
    --optimizer "$optimizer" \
    # --learning_rate 0.01 \
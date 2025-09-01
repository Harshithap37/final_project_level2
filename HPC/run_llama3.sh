#!/bin/bash
#SBATCH --job-name=llama3_infer
#SBATCH --output=../Output/llama3_output.txt
#SBATCH --error=../Output/llama3_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

# Load Anaconda and activate your conda environment
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate finalproj

# Navigate to the Code directory where your Python file lives
cd ../Code

# Run your LLaMA 3 inference script
python llama3_infer.py


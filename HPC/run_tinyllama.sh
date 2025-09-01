#!/bin/bash
#SBATCH --job-name=tinyllama_infer
#SBATCH --output=../Output/tinyllama_output.txt
#SBATCH --error=../Output/tinyllama_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

# Load Anaconda and activate your conda environment
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate finalproj

# Navigate to the Code directory where your Python file lives
cd ../Code

# Run your TinyLlama inference/training script
python tiny_llama.py

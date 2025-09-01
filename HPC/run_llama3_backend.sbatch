#!/bin/bash
#SBATCH --job-name=llama3_backend
#SBATCH --output=llama3_backend.out
#SBATCH --error=llama3_backend.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

# Load Anaconda
module load Anaconda3/2024.02-1
source ~/.bashrc

# Activate environment
conda activate finalproj

# Export your Hugging Face token
export HF_TOKEN="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"

# Navigate to the project folder
cd ~/Final_Project

# Run the backend server
python backend_hpc.py


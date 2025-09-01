#!/bin/bash
#SBATCH --job-name=combined_api
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=3
#SBATCH --time=02:00:00
#SBATCH --chdir=/mnt/parscratch/users/acp24hp/Final_Project_level2/HPC
#SBATCH --output=/mnt/parscratch/users/acp24hp/Final_Project_level2/Output/combined_api_out.txt
#SBATCH --error=/mnt/parscratch/users/acp24hp/Final_Project_level2/Output/combined_api_err.txt

# Fail fast, but allow sourcing bashrc without PS1 errors
set -eo pipefail
set +u
export PS1="${PS1:-}"
module load Anaconda3/2024.02-1
source ~/.bashrc
set -u

conda activate finalproj

# Make sure Output dir exists
mkdir -p /mnt/parscratch/users/acp24hp/Final_Project_level2/Output

# --- Hugging Face cache + token ---
unset HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE HUGGINGFACE_HUB_TOKEN HF_TOKEN
export HF_HOME=/mnt/parscratch/users/acp24hp/hf-cache
export HF_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

# Token saved by `hf auth login`
if [[ -f "$HOME/.cache/huggingface/token" ]]; then
  export HUGGINGFACE_HUB_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
  export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
else
  echo "ERROR: No Hugging Face token at $HOME/.cache/huggingface/token"
  echo "Run: hf auth login"
  exit 1
fi

# --- Model settings (optional overrides) ---
export MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
export MAX_NEW_TOKENS=400
export TEMPERATURE=0.6

# --- Start API ---
cd /mnt/parscratch/users/acp24hp/Final_Project_level2/Code

# Ensure deps (safe if already present)
python - <<'PY'
import sys, subprocess
for pkg in ["fastapi","uvicorn","transformers","torch","accelerate","pydantic","huggingface_hub","rank_bm25","sentence-transformers","faiss-cpu","pypdf","pypdfium2","python-docx","pytesseract","Pillow","z3-solver"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
print("Deps ok")
PY

# Clean any stale HF lock files (harmless if none)
find "$HF_HOME" -type f -name "*.lock" -delete 2>/dev/null || true

# Launch the *combined* FastAPI app on port 8001
python -m uvicorn combined_api:app --host 127.0.0.1 --port 8001

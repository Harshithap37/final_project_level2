# ---- build_index.py ----
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
# point to web/knowledge instead of Code/knowledge
KNOW_DIR = (BASE_DIR.parent / "web" / "knowledge").resolve()
INDEX_FILE = BASE_DIR / "faiss_index.bin"
META_FILE = BASE_DIR / "faiss_meta.pkl"

# Model
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_texts():
    docs = []
    meta = []
    for root, _, files in os.walk(KNOW_DIR):
        for fn in files:
            path = os.path.join(root, fn)
            if fn.lower().endswith(".pdf"):
                try:
                    reader = PdfReader(path)
                    text = " ".join([p.extract_text() or "" for p in reader.pages])
                except Exception as e:
                    print("PDF error", fn, e)
                    continue
            elif fn.lower().endswith(".txt"):
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except Exception as e:
                    print("TXT error", fn, e)
                    continue
            else:
                continue

            if text.strip():
                docs.append(text)
                meta.append({"file": fn, "path": path})
    return docs, meta

def main():
    docs, meta = load_texts()
    print(f"Loaded {len(docs)} documents from {KNOW_DIR}")

    if not docs:
        print("No documents found — check that your PDFs/TXT are in:", KNOW_DIR)
        return

    embeddings = model.encode(docs, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "wb") as f:
        pickle.dump(meta, f)

    print(f"Index built with {len(docs)} documents and saved to {INDEX_FILE}")

if __name__ == "__main__":
    main()
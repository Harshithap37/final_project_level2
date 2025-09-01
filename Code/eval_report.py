# ---- eval_report.py (extended: supports chat runs AND proof runs) ----
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- optional readers (unchanged) --------
try:
    from docx import Document as _DocxDocument
except Exception:
    _DocxDocument = None
try:
    from pypdf import PdfReader as _PdfReader
except Exception:
    _PdfReader = None

# Optional semantic model (unchanged)
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _SEM_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    _SEM_MODEL = None

# ---------------- chat loaders (unchanged core) ----------------
def load_runs(run_dir: Path) -> pd.DataFrame:
    runs_path = run_dir / "runs.jsonl"
    if not runs_path.exists():
        raise FileNotFoundError(f"Could not find {runs_path}")
    rows = []
    with runs_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    df = pd.DataFrame(rows)
    if "rag" not in df.columns and "use_rag" in df.columns:
        df["rag"] = df["use_rag"].map(lambda x: "ON" if x else "OFF")
    if "rag" not in df.columns:
        df["rag"] = "OFF"
    for col, default in [
        ("qid", None),
        ("question", ""),
        ("reply", ""),
        ("latency_sec", math.nan),
        ("tokens_out", math.nan),
        ("mode", df["mode"][0] if "mode" in df.columns and len(df) else "dense"),
        ("hits", []),
    ]:
        if col not in df.columns:
            df[col] = default
    df["rag"] = df["rag"].astype(str).str.upper()
    df["latency_sec"] = pd.to_numeric(df["latency_sec"], errors="coerce")
    df["tokens_out"] = pd.to_numeric(df["tokens_out"], errors="coerce")
    if "hits" in df.columns:
        df["hits_joined"] = df["hits"].apply(
            lambda h: ", ".join(map(str, h)) if isinstance(h, list) else str(h)
        )
    return df

def ensure_human_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["accuracy", "completeness", "citations", "notes"]:
        if col not in df.columns:
            df[col] = "" if col == "notes" else pd.NA
    return df

# ---------------- chat auto-metrics (unchanged) ----------------
def compute_keyword_hit(row, topk=3):
    q = str(row.get("question","")).lower()
    r = str(row.get("reply","")).lower()
    toks = [t for t in q.replace("/", " ").split() if t.isalpha()]
    toks = toks[:topk] if toks else []
    return int(any(t in r for t in toks)) if toks else np.nan

def compute_semantic_sim_answer(row):
    if _SEM_MODEL is None:
        return np.nan
    q = str(row.get("question",""))
    r = str(row.get("reply",""))
    if not q or not r:
        return np.nan
    qv = _SEM_MODEL.encode(q, convert_to_tensor=True, normalize_embeddings=True)
    rv = _SEM_MODEL.encode(r, convert_to_tensor=True, normalize_embeddings=True)
    sim = float(st_util.cos_sim(qv, rv).item())
    return sim

def add_basic_auto_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df["kw_hit"]   = df.apply(compute_keyword_hit, axis=1)
    df["sem_sim"]  = df.apply(compute_semantic_sim_answer, axis=1)
    return df

# ---------------- retrieval content loader (unchanged) ----------------
SUPPORTED_EXTS = {".txt", ".md", ".docx", ".csv", ".pdf"}

def _read_text(path: Path, max_chars: int = 1500) -> str:
    if not path.exists() or not path.is_file():
        return ""
    ext = path.suffix.lower()
    try:
        if ext in (".txt", ".md"):
            txt = path.read_text(encoding="utf-8", errors="ignore")
        elif ext == ".csv":
            import pandas as pd
            try: df = pd.read_csv(path)
            except Exception: df = pd.read_csv(path, sep=None, engine="python")
            txt = df.head(200).to_csv(index=False)
        elif ext == ".docx":
            if _DocxDocument is None:
                return f"[DOCX:{path.name}]"
            try:
                d = _DocxDocument(str(path))
                parts = []
                for p in d.paragraphs:
                    if p.text.strip(): parts.append(p.text.strip())
                for tbl in d.tables:
                    for row in tbl.rows:
                        row_txt = " | ".join((c.text or "").strip() for c in row.cells)
                        if row_txt.strip(): parts.append(row_txt)
                txt = "\n".join(parts).strip() or f"[DOCX:{path.name}]"
            except Exception:
                txt = f"[DOCX:{path.name}]"
        elif ext == ".pdf":
            if _PdfReader is None:
                return f"[PDF:{path.name}]"
            try:
                reader = _PdfReader(str(path))
                chunks = [(page.extract_text() or "") for page in reader.pages[:20]]
                txt = "\n".join(chunks).strip() or f"[PDF:{path.name}]"
            except Exception:
                txt = f"[PDF:{path.name}]"
        else:
            txt = ""
    except Exception:
        txt = ""
    if max_chars and len(txt) > max_chars:
        txt = txt[:max_chars] + " …"
    return txt

# ---------------- semantic retrieval metrics (unchanged) ----------------
def compute_semantic_retrieval_scores(row, knowledge_dir: Path) -> tuple[float, float, str]:
    if _SEM_MODEL is None:
        return np.nan, np.nan, "[]"
    hits = row.get("hits", [])
    if not isinstance(hits, list) or not hits:
        return np.nan, np.nan, "[]"
    q = str(row.get("question","")).strip()
    if not q:
        return np.nan, np.nan, "[]"
    qv = _SEM_MODEL.encode(q, convert_to_tensor=True, normalize_embeddings=True)
    sims = []
    for fname in hits:
        if not isinstance(fname, str): continue
        ext = Path(fname).suffix.lower()
        if ext and ext not in SUPPORTED_EXTS: continue
        fpath = Path(knowledge_dir) / fname
        if not fpath.exists():
            matches = list(Path(knowledge_dir).glob(fname)) or \
                      [p for p in Path(knowledge_dir).glob("*") if p.name.lower()==fname.lower()]
            if matches: fpath = matches[0]
            else: continue
        content = _read_text(fpath, max_chars=1500)
        cv = _SEM_MODEL.encode(content, convert_to_tensor=True, normalize_embeddings=True)
        sim = float(st_util.cos_sim(qv, cv).item())
        sims.append({"file": fpath.name, "sim": sim})
    if not sims:
        return np.nan, np.nan, "[]"
    max_sim = max(s["sim"] for s in sims)
    avg_sim = sum(s["sim"] for s in sims) / len(sims)
    import json as _json
    return max_sim, avg_sim, _json.dumps(sims, ensure_ascii=False)

def add_semantic_retrieval_metrics(df: pd.DataFrame, knowledge_dir: Path) -> pd.DataFrame:
    max_list, avg_list, per_doc_json = [], [], []
    for _, row in df.iterrows():
        m, a, js = compute_semantic_retrieval_scores(row, knowledge_dir)
        max_list.append(m); avg_list.append(a); per_doc_json.append(js)
    df["sem_retrieval_max"] = max_list
    df["sem_retrieval_avg"] = avg_list
    df["sem_retrieval_topk"] = per_doc_json
    return df

# ---------------- outputs ----------------
def write_tables(df: pd.DataFrame, out_path: Path) -> Path:
    df.to_csv(out_path, index=False)
    return out_path

def bar_plot(values: dict, title: str, ylabel: str, out_path: Path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5.2, 3.8), dpi=150)
    xs = list(values.keys())
    ys = list(values.values())
    plt.bar(xs, ys)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ---------------- proof-run loader ----------------
def load_proofs(proof_run_dir: Path) -> pd.DataFrame:
    p = proof_run_dir / "proofs.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"Could not find {p}")
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    df = pd.DataFrame(rows)
    # Normalize
    for col in ["tool","goal","code","error_type"]:
        if col not in df.columns: df[col] = ""
    if "proof_success" not in df.columns: df["proof_success"] = pd.NA
    if "latency_sec" in df.columns:
        df["latency_sec"] = pd.to_numeric(df["latency_sec"], errors="coerce")
    return df

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(
        description="Summarize chat runs and/or proof runs; compute metrics; save CSV + charts."
    )
    ap.add_argument("--run_dir", default=None,
                    help="Path to eval_runs/<timestamp> (chat). If omitted, skip chat summary.")
    ap.add_argument("--proof_run_dir", default=None,
                    help="Path to eval_proofs_runs/<timestamp> (proofs). If omitted, skip proof summary.")
    ap.add_argument("--knowledge_dir", default="knowledge",
                    help="Path to backend knowledge/ for retrieval quality (chat only).")
    args = ap.parse_args()

    # ---------- CHAT SUMMARY (optional) ----------
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        df = load_runs(run_dir)
        df = ensure_human_columns(df)
        df = add_basic_auto_metrics(df)
        df = add_semantic_retrieval_metrics(df, Path(args.knowledge_dir).expanduser().resolve())

        # Averages by RAG
        avg = df.groupby("rag", dropna=False).agg(
            avg_latency_sec=("latency_sec", "mean"),
            avg_tokens=("tokens_out", "mean"),
            n=("qid", "count"),
            kw_hit_rate=("kw_hit", "mean"),
            sem_sim_avg=("sem_sim", "mean"),
            sem_retr_max=("sem_retrieval_max", "mean"),
            sem_retr_avg=("sem_retrieval_avg", "mean"),
        )

        out_csv = run_dir / "report_chat.csv"
        write_tables(df, out_csv)

        # Charts
        avg_off = float(avg.loc["OFF", "avg_latency_sec"]) if "OFF" in avg.index else float("nan")
        avg_on  = float(avg.loc["ON",  "avg_latency_sec"])  if "ON"  in avg.index else float("nan")
        tok_off = float(avg.loc["OFF", "avg_tokens"])       if "OFF" in avg.index else float("nan")
        tok_on  = float(avg.loc["ON",  "avg_tokens"])       if "ON"  in avg.index else float("nan")

        bar_plot({"RAG OFF": avg_off, "RAG ON": avg_on}, "Average Latency (chat)", "seconds", run_dir / "latency_bar_chat.png")
        bar_plot({"RAG OFF": tok_off, "RAG ON": tok_on}, "Average Tokens (chat)", "tokens", run_dir / "tokens_bar_chat.png")

        print("\n=== Chat Evaluation ===")
        print(f"Run dir:  {run_dir}")
        print(f"Rows:     {len(df)}")
        print("Averages (by RAG):")
        shown_cols = ["avg_latency_sec","avg_tokens","kw_hit_rate","sem_sim_avg","sem_retr_max","sem_retr_avg"]
        print(avg[shown_cols])
        print(f"Saved CSV: {out_csv}")

    # ---------- PROOF SUMMARY (optional) ----------
    if args.proof_run_dir:
        pr_dir = Path(args.proof_run_dir).expanduser().resolve()
        pdf = load_proofs(pr_dir)

        # Success rate per tool
        grp = pdf.groupby("tool", dropna=False)
        summary = grp.agg(
            n=("tool","count"),
            success_rate=("proof_success", lambda s: float(np.nanmean([1 if x is True else 0 for x in s])) if len(s)>0 else float("nan")),
            avg_latency=("latency_sec","mean"),
        )
        out_csv_proofs = pr_dir / "report_proofs.csv"
        write_tables(pdf, out_csv_proofs)

        # Chart: proof success by tool
        # Convert success_rate NaNs to 0 for plotting clarity
        plot_vals = {idx: (0.0 if (pd.isna(row["success_rate"])) else float(row["success_rate"])) for idx, row in summary.iterrows()}
        bar_plot(plot_vals, "Proof Success Rate by Tool", "success rate (0–1)", pr_dir / "proof_success_by_tool.png")

        print("\n=== Proof Evaluation ===")
        print(f"Run dir:  {pr_dir}")
        print(f"Rows:     {len(pdf)}")
        print("Per-tool summary:")
        print(summary)
        print(f"Saved CSV: {out_csv_proofs}")

    if not args.run_dir and not args.proof_run_dir:
        print("Nothing to do. Provide --run_dir and/or --proof_run_dir.", file=sys.stderr)

if __name__ == "__main__":
    main()
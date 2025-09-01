# ---- eval_client.py ----
import os, json, time, argparse, pathlib
import requests

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8001")

def load_questions(path):
    """
    Returns a list of dicts: {"id": <custom id or None>, "question": <text>}.
    Supports .jsonl with {"id":..., "question":...} or .txt (one question per line).
    """
    qs = []
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                qtext = obj.get("question") or obj.get("q") or obj.get("text")
                if qtext:
                    qs.append({"id": obj.get("id"), "question": qtext})
        else:
            for line in f:
                qtext = line.strip()
                if qtext:
                    qs.append({"id": None, "question": qtext})
    return qs

def ask(question_text, use_rag: bool, retrieval_mode="hybrid", temperature=0.2, max_new_tokens=500):
    t0 = time.time()
    r = requests.post(f"{BACKEND}/chat", json={
        "message": question_text,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "use_rag": use_rag,
        "retrieval_mode": retrieval_mode,  # correct key for backend
    }, timeout=300)
    dt = time.time() - t0
    r.raise_for_status()
    js = r.json()
    reply = (js.get("reply") or "").strip()
    hits  = js.get("hits", [])            # used by eval_report for retrieval quality
    tokens_out = max(1, len(reply) // 4)  # crude proxy
    return reply, dt, tokens_out, hits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True, help="Path to questions.jsonl or .txt")
    ap.add_argument("--outdir", default="eval_runs", help="Where to save runs")
    ap.add_argument("--max_new_tokens", type=int, default=500)
    ap.add_argument("--mode", choices=["dense","bm25","hybrid"], default="hybrid",
                    help="Retrieval mode to request from backend")
    args = ap.parse_args()

    qs = load_questions(args.questions)
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = pathlib.Path(args.outdir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "runs.jsonl"

    print(f"Loaded {len(qs)} questions. Writing: {out_path}")

    with open(out_path, "w", encoding="utf-8") as out:
        for qid, qobj in enumerate(qs):
            qtext = qobj["question"]
            qcustom = qobj["id"]
            for use_rag in (False, True):
                reply, latency, toks, hits = ask(
                    qtext, use_rag=use_rag, retrieval_mode=args.mode, max_new_tokens=args.max_new_tokens
                )
                rec = {
                    "run_id": ts,
                    "qid": qid,                 # internal sequential id
                    "qid_custom": qcustom,      # your original id from JSONL (may be None)
                    "question": qtext,
                    "use_rag": use_rag,
                    "mode": args.mode,
                    "reply": reply,
                    "latency_sec": round(latency, 3),
                    "tokens_out": toks,
                    "hits": hits,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"Q{qid:03d} RAG={'ON' if use_rag else 'OFF'}  "
                      f"{args.mode.upper():6s}  lat={latency:.2f}s  toks≈{toks}")

    print("\nDone.")
    print(f"Next: python eval_report.py --run_dir {run_dir}")

if __name__ == "__main__":
    main()
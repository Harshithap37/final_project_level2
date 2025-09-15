import os, json, time, argparse, pathlib, tempfile, subprocess, sys

PROOF_API = os.getenv("PROOF_BACKEND_URL", "http://127.0.0.1:8001/proofapi")

import requests

def load_tasks(path):

    p = pathlib.Path(path)
    items = []
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    items.append(json.loads(s))
    else:
        items = json.loads(p.read_text(encoding="utf-8"))
    return items

def _local_check_coq(code: str, timeout: int = 15) -> tuple[bool,str]:
    try:
        subprocess.run(["coqc", "-v"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return (None, "unchecked")  
    with tempfile.NamedTemporaryFile("w", suffix=".v", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        proc = subprocess.run(["coqc", path],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=timeout)
        ok = (proc.returncode == 0)
        return (ok, "ok" if ok else "syntax_or_incomplete")
    except subprocess.TimeoutExpired:
        return (False, "timeout")
    finally:
        try: os.unlink(path)
        except Exception: pass

def _local_check_isabelle(code: str, timeout: int = 20) -> tuple[bool,str]:
    try:
        subprocess.run(["isabelle", "version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return (None, "unchecked")  
    with tempfile.NamedTemporaryFile("w", suffix=".thy", delete=False) as f:
        f.write(code)
        path = f.name
   
    try:
        proc = subprocess.run(["isabelle", "process", "-e", f"use_thy \"{path}\""],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=timeout)
        ok = (proc.returncode == 0)
        return (ok, "ok" if ok else "fail")
    except subprocess.TimeoutExpired:
        return (False, "timeout")
    finally:
        try: os.unlink(path)
        except Exception: pass

def prove_one(task, temperature=0.2, max_new_tokens=300, timeout_sec=8,
              local_check=False):
    payload = {
        "tool": task.get("tool","z3py").lower(),
        "goal": task["goal"],
        "context": task.get("context","") or "",
        "assumptions": task.get("assumptions") or [],
        "temperature": float(task.get("temperature", temperature)),
        "max_new_tokens": int(task.get("max_new_tokens", max_new_tokens)),
        "timeout_sec": int(task.get("timeout_sec", timeout_sec)),
    }
    t0 = time.time()
    r = requests.post(f"{PROOF_API}/prove", json=payload, timeout=timeout_sec+60)
    dt = time.time() - t0
    r.raise_for_status()
    js = r.json()

    code = js.get("code") or ""
    tool = js.get("tool") or payload["tool"]

    
    proof_success, error_type = None, "unchecked"

    
    if tool == "z3py" and "exec" in js:
        ex = js["exec"] or {}
        ok = bool(ex.get("ok"))
        proof_success = ok
        if ok:
            error_type = "ok"
        else:
            err = (ex.get("stderr") or "").lower()
            if "timeout" in err:
                error_type = "timeout"
            elif "syntax" in err or "indentation" in err:
                error_type = "syntax"
            else:
                error_type = "runtime"

   
    if local_check and proof_success is None:
        if tool == "coq":
            proof_success, error_type = _local_check_coq(code)
        elif tool == "isabelle":
            proof_success, error_type = _local_check_isabelle(code)

    return {
        "id": task.get("id"),
        "tool": tool,
        "goal": payload["goal"],
        "context": payload["context"],
        "assumptions": payload["assumptions"],
        "temperature": payload["temperature"],
        "max_new_tokens": payload["max_new_tokens"],
        "timeout_sec": payload["timeout_sec"],
        "latency_sec": round(dt, 3),
        "code": code,
        "exec": js.get("exec"),        
        "proof_success": proof_success,  
        "error_type": error_type,        
    }

def main():
    ap = argparse.ArgumentParser(description="Evaluate proof generation via /proofapi/prove.")
    ap.add_argument("--tasks", required=True, help="Path to tasks.jsonl or .json")
    ap.add_argument("--outdir", default="eval_proofs_runs", help="Output dir")
    ap.add_argument("--local_check", action="store_true",
                    help="Optionally validate Coq/Isabelle locally (requires tools installed)")
    args = ap.parse_args()

    tasks = load_tasks(args.tasks)
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = pathlib.Path(args.outdir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "proofs.jsonl"

    print(f"Loaded {len(tasks)} tasks. Writing: {out_path}")
    with out_path.open("w", encoding="utf-8") as out:
        for i, t in enumerate(tasks):
            try:
                rec = prove_one(t, local_check=args.local_check)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"{i:03d} {rec['tool']:9s}  lat={rec['latency_sec']:.2f}s  "
                      f"success={rec['proof_success']}  err={rec['error_type']}")
            except Exception as e:
                errrec = {
                    "id": t.get("id"),
                    "tool": t.get("tool"),
                    "goal": t.get("goal"),
                    "latency_sec": None,
                    "code": "",
                    "exec": None,
                    "proof_success": False,
                    "error_type": f"client_error:{e}",
                }
                out.write(json.dumps(errrec, ensure_ascii=False) + "\n")
                print(f"{i:03d} ERROR  {t.get('tool')}  {e}")

    print("\nDone.")
    print(f"Next: python eval_report.py --proof_run_dir {run_dir}")

if __name__ == "__main__":
    main()

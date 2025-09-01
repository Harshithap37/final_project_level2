# ---- proof_api.py — Level 2: Proof Assistant Bridge (codegen + optional Z3 run + logging) ----
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import re, os, json, subprocess, sys, tempfile, textwrap, shlex, resource, signal, time
import torch  # keep near top for clarity

# Reuse the already-loaded LLaMA-3 model to avoid double VRAM/CPU
# (Make sure proof_api.py sits in the same folder as llama3_api.py)
from llama3_api import tokenizer, model, device  # noqa: E402

# --------------------------- FastAPI app ---------------------------
app = FastAPI(title="Proof Bridge API (Level 2)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------- Logging (NEW) ---------------------------
LOG_BASE = os.path.join(os.path.dirname(__file__), "eval_proofs_runs", "weblog")
os.makedirs(LOG_BASE, exist_ok=True)
LOG_PATH = os.path.join(LOG_BASE, "proofs.jsonl")

def _log_proof(rec: dict):
    """Append one JSONL record of a proof request/response."""
    rec = dict(rec)
    rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# --------------------------- Schemas ---------------------------
class ProveIn(BaseModel):
    tool: str                 # "coq" | "isabelle" | "z3py" | "z3py_run"
    goal: str                 # natural language objective / property
    context: str | None = ""  # optional domain context or spec text
    assumptions: list[str] | None = None
    max_new_tokens: int | None = 300
    temperature: float | None = 0.2
    timeout_sec: int | None = 8  # for z3py_run

# --------------------------- Helpers ---------------------------
def _strip_code_fence(s: str) -> str:
    """Extract code from markdown fences if present."""
    fence = re.search(r"```(?:[a-zA-Z0-9_+-]*)\s*(.*?)```", s, flags=re.S)
    if fence:
        return fence.group(1).strip()
    return s.strip()

def _gen_with_llama(sys_prompt: str, user_prompt: str, max_new_tokens: int, temperature: float) -> str:
    msgs = [{"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_prompt}]
    input_ids = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0][input_ids.shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

def _tool_block_header(tool: str) -> str:
    if tool == "coq":
        return "```coq"
    if tool == "isabelle":
        return "```isabelle"
    # default z3py (python)
    return "```python"

def _build_codegen_prompt(tool: str, goal: str, context: str, assumptions: list[str] | None) -> tuple[str, str]:
    sys_prompt = (
        "You translate natural-language verification goals into minimal, correct snippets for proof tools.\n"
        "Tools supported: Coq (Gallina), Isabelle/Isar theory snippet, Z3 (python/z3-solver).\n"
        "Rules:\n"
        "• Emit ONLY a single fenced code block for the requested tool; no prose.\n"
        "• Keep it minimal and self-contained. Add small comments where clarifying.\n"
        "• If the user tool is 'z3py', use Python with z3-solver and print satisfiability and a model when SAT.\n"
        "• If assumptions are given, encode them as constraints/axioms.\n"
    )

    header = _tool_block_header(tool)
    footer = "```"

    user = []
    user.append(f"TOOL: {tool}")
    user.append(f"GOAL:\n{goal.strip()}")
    if context and context.strip():
        user.append(f"CONTEXT:\n{context.strip()}")
    if assumptions:
        user.append("ASSUMPTIONS:")
        for a in assumptions:
            user.append(f"- {a}")
    user.append("\nOutput format:\n"
                f"{header}\n"
                f"...code here...\n"
                f"{footer}")

    return sys_prompt, "\n".join(user)

# --------------------------- Optional Z3 sandbox ---------------------------
def _limit_resources(mem_mb=320, cpu_sec=5):
    """Apply soft resource limits for the current process (Linux/Unix)."""
    resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec))

def _run_z3py(code: str, timeout_sec: int = 8) -> dict:
    """
    Run z3py code in a restricted subprocess.

    IMPORTANT: Build the wrapper with NO leading indentation and indent the
    user code under 'try:' to avoid IndentationError on line 1.
    """
    # Indent user code under the try: block
    user_code = textwrap.indent(code.rstrip() + "\n", "    ")

    # Column-0 wrapper (no leading spaces)
    wrapper = (
        "import sys, json, traceback, signal, resource, os\n"
        "def _limit():\n"
        "    resource.setrlimit(resource.RLIMIT_AS, (335544320, 335544320))\n"
        "    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))\n"
        "_limit()\n"
        "try:\n"
        f"{user_code}"
        "except Exception as e:\n"
        "    print('RUNTIME_ERROR:', e, file=sys.stderr)\n"
        "    traceback.print_exc()\n"
        "    sys.exit(2)\n"
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(wrapper)
        path = f.name

    env = os.environ.copy()
    cmd = [sys.executable, "-I", path]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout_sec,
            env=env,
        )
        elapsed = round(time.time() - t0, 3)
        return {
            "ok": (proc.returncode == 0),
            "exit_code": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "time_sec": elapsed,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "exit_code": -1,
            "stdout": (e.stdout or "").strip() if isinstance(e.stdout, str) else "",
            "stderr": "TIMEOUT",
            "time_sec": timeout_sec,
        }
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass

# --------------------------- Routes ---------------------------
@app.post("/prove")
def prove(inp: ProveIn):
    t0 = time.time()

    tool = (inp.tool or "").lower().strip()
    if tool not in ("coq", "isabelle", "z3py", "z3py_run"):
        return {"ok": False, "error": "tool must be one of: coq | isabelle | z3py | z3py_run"}

    sys_prompt, user_prompt = _build_codegen_prompt(
        tool=("z3py" if tool == "z3py_run" else tool),
        goal=inp.goal,
        context=inp.context or "",
        assumptions=inp.assumptions or [],
    )

    code_raw = _gen_with_llama(
        sys_prompt=sys_prompt,
        user_prompt=user_prompt,
        max_new_tokens=int(inp.max_new_tokens or 300),
        temperature=float(inp.temperature if inp.temperature is not None else 0.2)
    )
    code = _strip_code_fence(code_raw)

    # Defaults for logging
    exec_res = None
    proof_success = None
    error_type = "not_executed" if tool in ("coq", "isabelle", "z3py") else None

    # Optionally run Z3
    if tool == "z3py_run":
        exec_res = _run_z3py(code, timeout_sec=int(inp.timeout_sec or 8))
        proof_success = bool(exec_res.get("ok"))
        if not proof_success:
            if exec_res.get("stderr") == "TIMEOUT" or exec_res.get("exit_code") == -1:
                error_type = "timeout"
            else:
                error_type = "runtime_error"
        else:
            error_type = "ok"

    latency = round(time.time() - t0, 3)

    # ---- LOG the request/response
    _log_proof({
        "tool": tool,                       # "coq" | "isabelle" | "z3py" | "z3py_run"
        "goal": inp.goal,
        "context": inp.context or "",
        "assumptions": inp.assumptions or [],
        "temperature": float(inp.temperature if inp.temperature is not None else 0.2),
        "max_new_tokens": int(inp.max_new_tokens or 300),
        "timeout_sec": int(inp.timeout_sec or 8),
        "latency_sec": latency,
        "code": code,
        "exec": exec_res,                   # None for non-run tools
        "proof_success": proof_success,     # True/False for z3py_run, else None
        "error_type": error_type            # "ok" | "timeout" | "runtime_error" | "not_executed"
    })

    # ---- Response back to caller
    if tool == "z3py_run":
        return {
            "ok": True,
            "tool": "z3py",
            "code": code,
            "exec": exec_res,
            "latency_sec": latency
        }

    return {
        "ok": True,
        "tool": tool,
        "code": code,
        "latency_sec": latency
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_device": str(device),
        "can_generate": True,
        "z3_runtime": "python z3-solver via sandboxed subprocess",
    }
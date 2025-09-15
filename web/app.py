from flask import Flask, render_template, request, jsonify
import requests, os

app = Flask(__name__)

CHAT_BACKEND  = os.getenv("CHAT_BACKEND_URL",  "http://127.0.0.1:8001/chatapi")
PROOF_BACKEND = os.getenv("PROOF_BACKEND_URL", "http://127.0.0.1:8001/proofapi")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/learn")
def learn():
    return render_template("learn.html")

@app.route("/resources")
def resources():
    return render_template("resources.html")

@app.route("/tools")
def tools():
    return render_template("tools.html")

@app.route("/metrics_page")
def metrics_page():
    return render_template("metrics.html")

@app.route("/upload_page")
def upload_page():
    return render_template("upload.html")



@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True) or {}

    payload = {
        "message":        str(data.get("message", "")).strip(),
        "temperature":    float(data.get("temperature", 0.20)),
        "use_rag":        bool(data.get("use_rag", True)),
        "retrieval_mode": (data.get("retrieval_mode") or "hybrid").strip().lower(),
        "max_new_tokens": min(int(data.get("max_new_tokens", 2000)), 4000),
        "mode":           (data.get("mode") or "simple").strip().lower(),
    }

    try:
        r = requests.post(f"{CHAT_BACKEND}/chat", json=payload, timeout=180)
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": f"Chat backend error: {e}"}), 500


@app.route("/health")
def health():
    # Return both services' health so the UI can show status for each
    try:
        r1 = requests.get(f"{CHAT_BACKEND}/health", timeout=5)
        chat = r1.json()
    except Exception as e:
        chat = {"ok": False, "error": f"chat: {e}"}

    try:
        r2 = requests.get(f"{PROOF_BACKEND}/health", timeout=5)
        proof = r2.json()
    except Exception as e:
        proof = {"ok": False, "error": f"proof: {e}"}

    return jsonify({"chat": chat, "proof": proof})


@app.route("/metrics")
def metrics():
    try:
        r = requests.get(f"{CHAT_BACKEND}/metrics", timeout=5)
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502


@app.route("/upload", methods=["POST"])
def upload():
    try:
        f = request.files["file"]
    except Exception:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        r = requests.post(
            f"{CHAT_BACKEND}/upload",
            files={"file": (f.filename, f.stream, f.mimetype)},
            timeout=60
        )
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": f"Upload backend error: {e}"}), 500


@app.route("/reindex", methods=["POST"])
def reindex():
    try:
        r = requests.post(f"{CHAT_BACKEND}/reindex", timeout=120)
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": f"Reindex backend error: {e}"}), 500



@app.route("/prove", methods=["POST"])
def prove():
    data = request.get_json(force=True) or {}

    tool = (data.get("tool") or "coq").strip().lower()
    goal = str(data.get("goal", "")).strip()
    if not goal:
        return jsonify({"error": "Field 'goal' is required"}), 400

    payload = {
        "tool": tool,                                 # "coq" | "isabelle" | "z3py" | "z3py_run"
        "goal": goal,                                 # natural-language objective
        "context": str(data.get("context", "") or ""),# optional domain/spec text
        "assumptions": data.get("assumptions") or [], # list[str]
        "max_new_tokens": int(data.get("max_new_tokens", 300)),
        "temperature": float(data.get("temperature", 0.2)),
        "timeout_sec": int(data.get("timeout_sec", 8)),  # used when tool == "z3py_run"
    }

    try:
        r = requests.post(f"{PROOF_BACKEND}/prove", json=payload, timeout=180)
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": f"Proof backend error: {e}"}), 500


if __name__ == "__main__":
    # overrides
    # CHAT_BACKEND_URL=http://127.0.0.1:8000 PROOF_BACKEND_URL=http://127.0.0.1:8001 python web/app.py
    app.run(host="127.0.0.1", port=5050, debug=True)
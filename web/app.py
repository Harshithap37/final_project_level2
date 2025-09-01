# ---- web/app.py ----
from flask import Flask, render_template, request, jsonify
import requests, os

app = Flask(__name__)

# Point to your FastAPI backend (can be overridden by env var)
BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8001")

# ---------- Page routes ----------
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


# ---------- API routes ----------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True) or {}

    # Pull values from client with sensible defaults
    message         = str(data.get("message", "")).strip()
    temperature     = float(data.get("temperature", 0.20))
    use_rag         = bool(data.get("use_rag", True))
    retrieval_mode  = (data.get("retrieval_mode") or "hybrid").strip().lower()
    max_new_tokens  = int(data.get("max_new_tokens", 2000))
    mode            = (data.get("mode") or "simple").strip().lower()

    # Keep a safety cap to avoid runaway generations
    MAX_CAP = 4000
    if max_new_tokens > MAX_CAP:
        max_new_tokens = MAX_CAP

    payload = {
        "message":        message,
        "temperature":    temperature,
        "use_rag":        use_rag,
        "retrieval_mode": retrieval_mode,   # <-- forward to FastAPI
        "max_new_tokens": max_new_tokens,
        "mode":           mode,
    }

    try:
        r = requests.post(f"{BACKEND}/chat", json=payload, timeout=180)
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": f"Chat backend error: {e}"}), 500


@app.route("/health")
def health():
    """Proxy to FastAPI /health so the UI status dot works."""
    try:
        r = requests.get(f"{BACKEND}/health", timeout=5)
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502


@app.route("/metrics")
def metrics():
    """Proxy to FastAPI /metrics (used by metrics.html)."""
    try:
        r = requests.get(f"{BACKEND}/metrics", timeout=5)
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502


@app.route("/upload", methods=["POST"])
def upload():
    """Proxy file uploads to FastAPI /upload."""
    try:
        f = request.files["file"]
    except Exception:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        r = requests.post(
            f"{BACKEND}/upload",
            files={"file": (f.filename, f.stream, f.mimetype)},
            timeout=60
        )
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": f"Upload backend error: {e}"}), 500

@app.route("/reindex", methods=["POST"])
def reindex():
    """Proxy manual reindex requests to FastAPI /reindex."""
    try:
        r = requests.post(f"{BACKEND}/reindex", timeout=120)
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": f"Reindex backend error: {e}"}), 500
        
# ---------- Run locally ----------
if __name__ == "__main__":
    # Example: BACKEND_URL=http://127.0.0.1:8001 python web/app.py
    app.run(host="127.0.0.1", port=5050, debug=True)
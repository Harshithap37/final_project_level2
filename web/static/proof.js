(function () {
  const btn = document.getElementById("pa-run");
  if (!btn) return;

  const toolEl = document.getElementById("pa-tool");
  const goalEl = document.getElementById("pa-goal");
  const ctxEl  = document.getElementById("pa-context");
  const assEl  = document.getElementById("pa-assumptions");
  const tempEl = document.getElementById("pa-temp");
  const maxEl  = document.getElementById("pa-max");

  const statusEl = document.getElementById("pa-status");
  const resBox = document.getElementById("pa-result");
  const codeEl = document.getElementById("pa-code");
  const execBox = document.getElementById("pa-exec");
  const execStatus = document.getElementById("pa-exec-status");
  const execStdout = document.getElementById("pa-exec-stdout");
  const execStderr = document.getElementById("pa-exec-stderr");

  async function postJSON(url, body) {
    const r = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body)
    });
    if (!r.ok) {
      const txt = await r.text().catch(() => "");
      throw new Error(`HTTP ${r.status}: ${txt || r.statusText}`);
    }
    return r.json();
  }

  btn.addEventListener("click", async () => {
    const tool = (toolEl.value || "coq").trim().toLowerCase();
    const goal = (goalEl.value || "").trim();
    const context = (ctxEl.value || "").trim();
    const assumptions = (assEl.value || "")
      .split("\n")
      .map(s => s.trim())
      .filter(Boolean);

    const temperature = Math.max(0, Math.min(1, parseFloat(tempEl.value || "0.2")));
    const max_new_tokens = Math.max(50, Math.min(1000, parseInt(maxEl.value || "300", 10) || 300));

    if (!goal) {
      statusEl.textContent = "Please enter a goal.";
      return;
    }

    statusEl.textContent = "Generating…";
    resBox.style.display = "none";
    execBox.style.display = "none";

    try {
      const payload = {
        tool,
        goal,
        context,
        assumptions,
        temperature,
        max_new_tokens,
        timeout_sec: 8
      };

      const out = await postJSON("/prove", payload);

      if (out.error) {
        statusEl.textContent = `Error: ${out.error}`;
        return;
      }

      codeEl.textContent = out.code || "";
      resBox.style.display = "block";

      if (tool === "z3py_run" && out.exec) {
        execStatus.textContent = `${out.exec.ok ? "ok" : "fail"} (exit=${out.exec.exit_code}, time=${out.exec.time_sec}s)`;
        execStdout.textContent = out.exec.stdout || "";
        execStderr.textContent = out.exec.stderr || "";
        execBox.style.display = "block";
      }

      statusEl.textContent = "Done.";
    } catch (e) {
      statusEl.textContent = `Error: ${e.message}`;
    }
  });
})();

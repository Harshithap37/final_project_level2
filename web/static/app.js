const chatLog     = document.getElementById("chat-log");
const chatForm    = document.getElementById("chat-form");
const chatInput   = document.getElementById("chat-input");
const sendBtn     = document.getElementById("send-btn");
const clearBtn    = document.getElementById("clear-btn");
const copyBtn     = document.getElementById("copy-last-btn");
const chips       = document.getElementById("chips");
const statusDot   = document.getElementById("status-dot");


const modeSel     = document.getElementById("mode");

const settingsBtn   = document.getElementById("settings-btn");
const settingsModal = document.getElementById("settings-modal");
const settingsClose = document.getElementById("settings-close");
const tempRange     = document.getElementById("temp-range");
const tempVal       = document.getElementById("temp-val");

const useRagCb = document.getElementById("use-rag");

const infoPanel = document.getElementById("info-panel");
const infoTitle = document.getElementById("info-title");
const infoBody  = document.getElementById("info-body");
document.getElementById("info-close")?.addEventListener("click", ()=> { if (infoPanel) infoPanel.hidden = true; });

let lastBotText = "";
let temperature = window.DEFAULT_TEMPERATURE ?? 0.2;

const getRetrievalMode = () => (window.RETRIEVAL_MODE || "hybrid");
const getMode = () => (modeSel?.value || "simple");

const esc = s => s.replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m]));
function mdLite(text){
  let out = esc(text);
  out = out.replace(/```([\s\S]*?)```/g, (_,code)=> `<pre><code>${code.trim()}</code></pre>`);
  out = out.replace(/`([^`]+)`/g, (_,code)=> `<code>${code}</code>`);
  out = out.replace(/(^|\n)- (.*)/g, (_,p,li)=> `${p}<div>• ${li}</div>`);
  return out;
}
function nowTime(){
  const d = new Date();
  const hh = String(d.getHours()).padStart(2,"0");
  const mm = String(d.getMinutes()).padStart(2,"0");
  return `${hh}:${mm}`;
}
function addRow(text, who, sources){
  const row = document.createElement("div");
  row.className = "rowmsg";

  const avatar = document.createElement("div");
  avatar.className = `avatar ${who}`;
  avatar.textContent = who === "user" ? "U" : "A";

  const wrap = document.createElement("div");
  wrap.className = "bubble-wrap";

  const bubble = document.createElement("div");
  bubble.className = `msg ${who}`;
  bubble.innerHTML = mdLite(text);

  if (Array.isArray(sources) && sources.length) {
    const cite = document.createElement("div");
    cite.className = "muted";
    cite.textContent = "Retrieved: " + sources.slice(0, 1).join(", ");
    bubble.appendChild(cite);
  }

  const ts = document.createElement("div");
  ts.className = "timestamp";
  ts.textContent = nowTime();

  wrap.append(bubble, ts);
  row.append(avatar, wrap);
  chatLog?.appendChild(row);
  if (chatLog) chatLog.scrollTop = chatLog.scrollHeight;
  return bubble;
}
function addTyping(){
  const row = document.createElement("div");
  row.className = "rowmsg";

  const av = document.createElement("div");
  av.className = "avatar bot";
  av.textContent = "A";

  const wrap = document.createElement("div");
  wrap.className = "bubble-wrap";

  const bubble = document.createElement("div");
  bubble.className = "msg bot";
  bubble.innerHTML = `<span class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>`;

  const ts = document.createElement("div");
  ts.className = "timestamp";
  ts.textContent = nowTime();

  wrap.append(bubble, ts);
  row.append(av, wrap);
  chatLog?.appendChild(row);
  if (chatLog) chatLog.scrollTop = chatLog.scrollHeight;
  return row;
}

async function healthPing(){
  try{
    const t0 = performance.now();
    const r = await fetch(window.STATUS_ENDPOINT, {cache:"no-store"});
    const js = await r.json();
    const t1 = performance.now();

    const okChat  = !!(js?.chat && (js.chat.status === "ok" || js.chat.ok === true));
    const okProof = !!(js?.proof && (js.proof.status === "ok" || js.proof.ok === true));
    const allGood = okChat && okProof;

    const lat = document.getElementById("latency");
    if (lat) lat.textContent = `${Math.round(t1 - t0)} ms`;

    statusDot?.classList.toggle("danger", !allGood);
    if (statusDot){
      statusDot.title = `chat: ${okChat ? "ok" : "down"} • proof: ${okProof ? "ok" : "down"}`;
    }
  }catch{
    statusDot?.classList.add("danger");
    const lat = document.getElementById("latency");
    if (lat) lat.textContent = "—";
    if (statusDot) statusDot.title = "backend unreachable";
  }
}
healthPing();
setInterval(healthPing, 15000);

if (chips){
  chips.addEventListener("click", e=>{
    const b = e.target.closest(".chip");
    if(!b) return;
    chatInput.value = b.textContent.trim();
    chatInput.focus();
  });
}

settingsBtn?.addEventListener("click", () => {
  if (!settingsModal) return;
  if (typeof settingsModal.showModal === "function") {
    settingsModal.showModal();
  } else {
    settingsModal.setAttribute("open", "");
  }
});
settingsClose?.addEventListener("click", () => {
  if (!settingsModal) return;
  if (typeof settingsModal.close === "function") {
    settingsModal.close();
  } else {
    settingsModal.removeAttribute("open");
  }
});
tempRange?.addEventListener("input", ()=>{
  if (!tempRange) return;
  if (tempVal) tempVal.textContent = Number(tempRange.value).toFixed(2);
  temperature = Number(tempRange.value);
});

chatForm?.addEventListener("submit", async (e)=>{
  e.preventDefault();
  const text = chatInput.value.trim();
  if(!text) return;

  addRow(text, "user");
  chatInput.value = "";
  if (sendBtn) sendBtn.disabled = true;

  const typingRow = addTyping();
  try{
    const r = await fetch(window.FLASK_ASK_ENDPOINT, {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({
        message: text,
        temperature,
        use_rag: useRagCb ? !!useRagCb.checked : true,
        retrieval_mode: getRetrievalMode(), 
        mode: getMode()                      
      })
    });
    const js = await r.json();
    typingRow.remove();
    if (r.ok && js.reply){
      lastBotText = js.reply;

      addRow(js.reply, "bot", js.hits);
    } else {
      lastBotText = "";
      addRow("Error: " + (js.error || "Backend unavailable"), "bot");
    }
  }catch(err){
    typingRow.remove();
    lastBotText = "";
    addRow("Network error contacting backend.", "bot");
  }finally{
    if (sendBtn) sendBtn.disabled = false;
  }
});

clearBtn?.addEventListener("click", ()=>{
  if (chatLog) chatLog.innerHTML = "";
  lastBotText = "";
});
copyBtn?.addEventListener("click", async ()=>{
  if(!lastBotText) return;
  try{
    await navigator.clipboard.writeText(lastBotText);
    if (copyBtn){
      copyBtn.textContent = "Copied!";
      setTimeout(()=> copyBtn.textContent = "Copy Reply", 1200);
    }
  }catch{}
});

(function initPolyhedron(){
  const canvas = document.getElementById("scene");
  if (!canvas || !window.THREE) return;

  const LABELS = [
    { name:"Coq",      desc:"Interactive prover (Gallina). Dependent types; extraction.", link:"https://coq.inria.fr/" },
    { name:"Isabelle", desc:"Generic HOL prover; locales; Sledgehammer.",                  link:"https://isabelle.in.tum.de/" },
    { name:"Lean",     desc:"HOL + dependent types; mathlib; tactics.",                    link:"https://leanprover-community.github.io/" },
    { name:"Agda",     desc:"Dependently-typed programming; proofs-as-programs.",         link:"https://agda.readthedocs.io/" },
    { name:"Z3",       desc:"SMT solver; theories; automation.",                           link:"https://github.com/Z3Prover/z3" },
    { name:"HOL4",     desc:"Classic higher-order logic prover.",                          link:"https://hol-theorem-prover.org/" },
  ];

  const renderer = new THREE.WebGLRenderer({ canvas, antialias:true, alpha:true });
  function size() {
    const r = canvas.getBoundingClientRect();
    renderer.setSize(r.width, r.height, false);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio||1, 2));
    camera.aspect = r.width / r.height;
    camera.updateProjectionMatrix();
  }

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  camera.position.set(0,0,6);
  size();
  window.addEventListener("resize", size);

  const group = new THREE.Group();
  scene.add(group);

  const geom = new THREE.IcosahedronGeometry(2.1, 1);
  const mat  = new THREE.MeshBasicMaterial({ color:0x7aa2ff, wireframe:true, transparent:true, opacity:0.35 });
  const mesh = new THREE.Mesh(geom, mat);
  group.add(mesh);

  function makeLabelSprite(text) {
    const padX = 26, padY = 16, fpx = 26;
    const m = document.createElement("canvas").getContext("2d");
    m.font = `600 ${fpx}px Inter, system-ui, sans-serif`;
    const w = Math.ceil(m.measureText(text).width) + padX * 2;
    const h = fpx + padY * 2;
    const c = document.createElement("canvas"); c.width=w; c.height=h;
    const ctx = c.getContext("2d");
    ctx.font = `600 ${fpx}px Inter, system-ui, sans-serif`;
    ctx.fillStyle = "rgba(215, 225, 255, 0.92)";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(text, w/2, h/2);
    const tex = new THREE.CanvasTexture(c);
    const mat = new THREE.SpriteMaterial({ map: tex, transparent:true });
    const sp  = new THREE.Sprite(mat);
    const aspect = w/h, base = 1.15;
    sp.scale.set(base, base/aspect, 1);
    return sp;
  }

  const positions = [
    [ 2.2,  0.6,  0], [-2.0,  0.5,  0],
    [ 0.0,  2.1,  0], [ 0.0, -2.1,  0],
    [ 1.8, -1.7,  0], [-1.8, -1.6,  0],
  ];

  const objs = LABELS.map((d,i)=>{
    const s = makeLabelSprite(d.name);
    const p = positions[i % positions.length];
    s.position.set(p[0], p[1], p[2]);
    group.add(s);
    return { sprite:s, data:d, baseX:s.scale.x, baseY:s.scale.y };
  });

  const tip = document.createElement("div");
  tip.className = "tooltip";
  tip.style.opacity = "0";
  document.body.appendChild(tip);

  const ray = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  let hovered = null;

  function onMove(e){
    const r = canvas.getBoundingClientRect();
    mouse.x = ((e.clientX - r.left)/r.width)*2 - 1;
    mouse.y = -((e.clientY - r.top)/r.height)*2 + 1;
  }
  window.addEventListener("mousemove", onMove);

  const tmpV = new THREE.Vector3();
  function setTip(obj) {
    const r = canvas.getBoundingClientRect();
    obj.getWorldPosition(tmpV).project(camera);
    const x = r.left + (tmpV.x*0.5+0.5)*r.width;
    const y = r.top  + (-tmpV.y*0.5+0.5)*r.height;
    tip.style.left = `${x + 10}px`;
    tip.style.top  = `${y - 10}px`;
  }

  canvas.addEventListener("click", ()=>{
    if (!infoPanel || !infoTitle || !infoBody) return;
    ray.setFromCamera(mouse, camera);
    const hits = ray.intersectObjects(objs.map(o=>o.sprite));
    if (!hits.length) return;
    const hit = objs.find(o=>o.sprite === hits[0].object);
    if (!hit) return;
    infoTitle.textContent = hit.data.name;
    infoBody.innerHTML = `
      <p>${esc(hit.data.desc)}</p>
      <p><a href="${hit.data.link}" target="_blank" rel="noopener">Official site ↗</a></p>
    `;
    infoPanel.hidden = false;
  });

  function tick(){
    group.rotation.y += 0.003;
    group.rotation.x += 0.0012;

    ray.setFromCamera(mouse, camera);
    const hits = ray.intersectObjects(objs.map(o=>o.sprite));
    if (hits.length){
      const sp = hits[0].object;
      if (hovered && hovered !== sp){
        const prev = objs.find(o=>o.sprite===hovered);
        hovered.scale.set(prev.baseX, prev.baseY, 1);
      }
      hovered = sp;
      const o = objs.find(x=>x.sprite===sp);
      sp.scale.set(o.baseX*1.1, o.baseY*1.1, 1);
      tip.textContent = o.data.desc;
      tip.style.opacity = "1";
      setTip(sp);
    }else{
      if (hovered){
        const o = objs.find(x=>x.sprite===hovered);
        hovered.scale.set(o.baseX, o.baseY, 1);
        hovered = null;
      }
      tip.style.opacity = "0";
    }

    renderer.render(scene, camera);
    requestAnimationFrame(tick);
  }
  tick();
})();
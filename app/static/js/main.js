/* ============================================================
   角色扮演 LLM 评测平台 — 通用交互逻辑
   ============================================================ */

const API = {
  loadModel: (data) => fetch('/model/load', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  }).then(r => r.json()),

  getRoles: () => fetch('/roles').then(r => r.json()).then(data => data.roles || []),

  chat: (data) => fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  }).then(r => r.json())
};

/* ---------- Nav scroll effect ---------- */
const nav = document.getElementById('mainNav');
if (nav) {
  window.addEventListener('scroll', () => {
    nav.classList.toggle('scrolled', window.scrollY > 10);
  }, { passive: true });
}

/* ---------- Loading overlay ---------- */
function showLoading(text) {
  const overlay = document.getElementById('loadingOverlay');
  if (!overlay) return;
  const txt = overlay.querySelector('.spinner-text');
  if (txt) txt.innerText = text || '处理中';
  overlay.classList.add('active');
}

function hideLoading() {
  const overlay = document.getElementById('loadingOverlay');
  if (overlay) overlay.classList.remove('active');
}

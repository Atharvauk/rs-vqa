"""
final_dashboard.py — FM-12a showcase dashboard (polished dark theme, 7B fine-tuned).

Features: dark UI with custom CSS; 7B fine-tuned model (qlora_adapter_7b);
tabs Demo|About|Results; a manual step-by-step coach tour (Next button, no
auto-advance) with a congrats message at the end; example questions; embedded
SVG accuracy chart. A single "Answer" button returns the answer AND always draws
the detected-object bounding boxes (detector runs on CPU so it does not compete
with the 7B for GPU memory).

Run (from FM12A):
    python final_dashboard.py --load-4bit
    # public link: --share    |    UI-only test: --mock
Open http://127.0.0.1:7860
"""
import argparse, sys, time
from pathlib import Path

BASE = Path(__file__).resolve().parent
TMP  = BASE / "_dash_tmp.png"
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

def adapter_7b():
    p = BASE / "7B" / "qlora_adapter_7b"
    return str(p if p.exists() else BASE / "qlora_adapter_7b")

DEMO_SYS = ("You are answering a question about an overhead satellite or aerial image. "
            "Give one clear, complete answer as a short word or phrase. Do not explain.")

RS_CLASSES = ("building. house. car. vehicle. ship. boat. airplane. plane. road. bridge. "
              "tree. storage tank. tennis court. swimming pool. roundabout. harbor. "
              "stadium. windmill. parking lot. runway. dam. ground track field.")

EXAMPLES = [
    "Is there a swimming pool in the image?",
    "What colour is the large building?",
    "Is this an urban or a rural area?",
    "What is the large object in the centre?",
    "How many ships are in the harbour?",
]

# ---------------------------------------------------------------- chart (inline SVG, no deps)
def bar_chart_svg():
    """Grouped bar chart: exact-match vs LLM-judge accuracy across configs."""
    data = [("3B zero-shot", 51.0, 60.8), ("3B fine-tuned", 51.5, 61.3),
            ("7B zero-shot", 51.3, 62.6), ("7B fine-tuned", 56.7, 66.7),
            ("API 35B", 42.8, 53.5)]
    W, H = 900, 380
    top, bottom, left = 30, 300, 55
    maxv = 75.0
    scale = (bottom - top) / maxv
    n = len(data)
    gw = (W - left - 20) / n
    bw = 30
    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
         f'style="width:100%;height:auto;font-family:Inter,Segoe UI,sans-serif">']
    for v in (0, 25, 50, 75):
        y = bottom - v * scale
        s.append(f'<line x1="{left}" y1="{y:.0f}" x2="{W-20}" y2="{y:.0f}" '
                 f'stroke="#24304d" stroke-width="1"/>')
        s.append(f'<text x="{left-10}" y="{y+4:.0f}" fill="#9fb0cc" font-size="12" '
                 f'text-anchor="end">{v}</text>')
    for i, (lab, ex, ju) in enumerate(data):
        cx = left + i * gw + gw / 2
        x1 = cx - bw - 3
        x2 = cx + 3
        h1, h2 = ex * scale, ju * scale
        best = lab == "7B fine-tuned"
        c1 = "#2dd4bf" if best else "#0e7490"
        c2 = "#99f6e4" if best else "#38bdf8"
        s.append(f'<rect x="{x1:.0f}" y="{bottom-h1:.0f}" width="{bw}" height="{h1:.0f}" rx="4" fill="{c1}"/>')
        s.append(f'<rect x="{x2:.0f}" y="{bottom-h2:.0f}" width="{bw}" height="{h2:.0f}" rx="4" fill="{c2}"/>')
        s.append(f'<text x="{x1+bw/2:.0f}" y="{bottom-h1-6:.0f}" fill="#e6edf7" font-size="11" text-anchor="middle">{ex:.1f}</text>')
        s.append(f'<text x="{x2+bw/2:.0f}" y="{bottom-h2-6:.0f}" fill="#e6edf7" font-size="11" text-anchor="middle">{ju:.1f}</text>')
        weight = "700" if best else "500"
        fill = "#2dd4bf" if best else "#cbd5e1"
        s.append(f'<text x="{cx:.0f}" y="{bottom+20:.0f}" fill="{fill}" font-size="12.5" '
                 f'font-weight="{weight}" text-anchor="middle">{lab}</text>')
    s.append(f'<rect x="{left}" y="8" width="13" height="13" rx="3" fill="#2dd4bf"/>')
    s.append(f'<text x="{left+18}" y="19" fill="#e6edf7" font-size="12.5">Exact-match</text>')
    s.append(f'<rect x="{left+130}" y="8" width="13" height="13" rx="3" fill="#38bdf8"/>')
    s.append(f'<text x="{left+148}" y="19" fill="#e6edf7" font-size="12.5">LLM-judge (lenient)</text>')
    s.append('</svg>')
    return "".join(s)


# ---------------------------------------------------------------- styled section HTML
ABOUT_HTML = """
<div class="sec">
  <h2 class="sec-h">🛰️ About this project</h2>
  <p class="sec-lead">
    <b>Remote-Sensing Visual Question Answering</b> — ask questions about objects in satellite and
    aerial images. The model is <b>Qwen2.5-VL-7B</b>, fine-tuned on the <b>VRSBench</b> benchmark with
    <b>QLoRA</b> and run <b>locally in 4-bit</b> (no paid cloud APIs in the answering path).
  </p>
  <p class="sec-sub">The project studies <i>reliability</i> — which engineering add-ons genuinely help:</p>
  <div class="card-grid-3">
    <div class="mini"><div class="tag tag-red">RQ1</div>
      <p>Injecting an object detector's evidence into the prompt <b>hurts</b> accuracy — the detector is
      trained on ordinary photos, not overhead imagery.</p></div>
    <div class="mini"><div class="tag tag-green">RQ2</div>
      <p>A rule-based <b>consistency check</b> that abstains on likely hallucinations <b>improves</b> reliability.</p></div>
    <div class="mini"><div class="tag tag-teal">Fine-tune</div>
      <p>Answer-only-masked QLoRA at a low learning rate <b>lifts</b> accuracy without breaking hard
      types like counting.</p></div>
  </div>
  <p class="sec-foot">ECS8060 — Artificial Intelligence Engineering · Queen's University Belfast ·
     Atharva Uday Kalase (40509608)</p>
</div>
"""

def results_html(chart):
    return f"""
<div class="sec">
  <h2 class="sec-h">📊 Results</h2>
  <p class="sec-lead">Accuracy on a fixed <b>1,000-question</b> VRSBench test set
     (exact-match / lenient LLM-judge).</p>
  <table class="res-table">
    <tr><th>Model</th><th>Zero-shot</th><th>Fine-tuned</th></tr>
    <tr><td>Qwen2.5-VL 3B</td><td>51.0% / 60.8%</td><td>51.5% / 61.3%</td></tr>
    <tr class="best"><td>Qwen2.5-VL 7B <span class="chip">this dashboard</span></td>
        <td>51.3% / 62.6%</td><td><b>56.7% / 66.7% ★</b></td></tr>
    <tr><td>Qwen3.6-35B (general, API)</td><td>42.8% / 53.5%</td><td>—</td></tr>
  </table>
  <div class="chart-wrap">{chart}</div>
  <div class="callout">
    <b>Takeaways:</b> grounding <i>hurts</i> (detector domain gap); a consistency check reduces
    hallucination; careful fine-tuning helps; and a small <i>specialised</i> model beats a much larger
    <i>general</i> one — <b>domain fit beats scale</b>.
  </div>
</div>
"""

# ---------------------------------------------------------------- interactive coach tour (JS)
# Manual walkthrough: the user advances with a Next button (no auto-advance);
# after the last step a short congrats message is shown.
TOUR_JS = """
() => {
  document.body.classList.add('dark');
  if (window.__fm12aTour) return;          // run once
  window.__fm12aTour = true;

  const steps = [
    {sel:'#in-image',    title:'Step 1 · Upload', text:'Upload a satellite / aerial image here.'},
    {sel:'#in-question', title:'Step 2 · Ask',    text:'Type your question here — or click an example chip below.'},
    {sel:'#answer-btn',  title:'Step 3 · Answer', text:'Click Answer to get the result and the detected objects.'}
  ];
  let i = 0, done = false;

  const ring = document.createElement('div'); ring.id = 'tour-ring'; ring.style.display = 'none';
  const tip  = document.createElement('div'); tip.id  = 'tour-tip';  tip.style.display  = 'none';
  document.body.appendChild(ring);
  document.body.appendChild(tip);

  const target = () => document.querySelector(steps[i].sel);

  function finish() {
    if (ring) ring.remove();
    if (tip) tip.remove();
    window.removeEventListener('resize', place);
    window.removeEventListener('scroll', place, true);
    if (window.__fm12aPoll) clearInterval(window.__fm12aPoll);
  }

  function congrats() {
    done = true;
    ring.style.display = 'none';
    tip.style.display = 'block';
    tip.style.maxWidth = '340px';
    tip.style.top  = Math.max(60, (window.innerHeight/2 - 95)) + 'px';
    tip.style.left = Math.max(14, (window.innerWidth/2 - 170)) + 'px';
    tip.innerHTML =
      '<div class="tt-h" style="font-size:14px">🎉 All set!</div>' +
      '<div class="tt-b">Nice work — you have completed the walkthrough. Ask anything about a ' +
      'satellite image and the fine-tuned 7B model will answer, with the detected objects shown ' +
      'alongside. Have fun exploring!</div>' +
      '<div class="tt-f"><span></span><button id="tour-next" class="tour-primary">Start exploring</button></div>';
    document.getElementById('tour-next').onclick = finish;
  }

  function place() {
    if (done) return;
    const el = target();
    if (!el) { ring.style.display = 'none'; tip.style.display = 'none'; return; }
    ring.style.display = 'block'; tip.style.display = 'block';
    const r = el.getBoundingClientRect(); const pad = 8;
    ring.style.top    = (r.top - pad) + 'px';
    ring.style.left   = (r.left - pad) + 'px';
    ring.style.width  = (r.width + pad*2) + 'px';
    ring.style.height = (r.height + pad*2) + 'px';
    let top = r.bottom + 14;
    if (top + 165 > window.innerHeight) top = Math.max(14, r.top - 165);
    let left = Math.min(Math.max(14, r.left), window.innerWidth - 320);
    tip.style.top = top + 'px'; tip.style.left = left + 'px';
    const last = (i === steps.length - 1);
    tip.innerHTML =
      '<div class="tt-h">' + steps[i].title + '</div>' +
      '<div class="tt-b">' + steps[i].text + '</div>' +
      '<div class="tt-f"><span>' + (i+1) + ' / ' + steps.length + '</span>' +
      '<span class="tt-btns">' +
      '<button id="tour-skip">Skip</button>' +
      '<button id="tour-next" class="tour-primary">' + (last ? 'Finish' : 'Next →') + '</button>' +
      '</span></div>';
    document.getElementById('tour-skip').onclick = finish;
    document.getElementById('tour-next').onclick = () => {
      if (i >= steps.length - 1) { congrats(); } else { i++; place(); }
    };
  }

  // keep the highlight glued to the current step's element (no auto-advance)
  window.__fm12aPoll = setInterval(() => { if (!done && target()) place(); }, 500);
  window.addEventListener('resize', place);
  window.addEventListener('scroll', place, true);
}
"""

# ---------------------------------------------------------------- custom dark theme
CSS = """
:root, .dark {
    --bg-0: #0b1120;
    --bg-1: #111a2e;
    --bg-2: #16213b;
    --stroke: #24304d;
    --teal: #2dd4bf;
    --teal-2: #14b8a6;
    --text: #e6edf7;
    --muted: #b7c4dc;
}
.gradio-container {
    background: radial-gradient(1200px 700px at 50% -14%, #14233f 0%, var(--bg-0) 55%) !important;
    color: var(--text) !important;
    max-width: 100% !important;
    padding: 18px 34px 40px 34px !important;
    font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
}
/* ===== interactive coach tour ===== */
#tour-ring {
    position: fixed; z-index: 9998; border: 3px solid #2dd4bf; border-radius: 14px;
    pointer-events: none; transition: top .25s, left .25s, width .25s, height .25s;
    box-shadow: 0 0 0 9999px rgba(4,10,22,.62), 0 0 22px rgba(45,212,191,.85);
    animation: tourpulse 1.6s ease-in-out infinite;
}
@keyframes tourpulse {
    0%,100% { box-shadow: 0 0 0 9999px rgba(4,10,22,.62), 0 0 16px rgba(45,212,191,.7); }
    50%     { box-shadow: 0 0 0 9999px rgba(4,10,22,.62), 0 0 34px rgba(45,212,191,1); }
}
#tour-tip {
    position: fixed; z-index: 9999; max-width: 300px;
    background: linear-gradient(135deg, #16213b, #111a2e); color: #e6edf7;
    border: 1px solid #2dd4bf; border-radius: 12px; padding: 12px 14px;
    font-family: Inter, Segoe UI, sans-serif; box-shadow: 0 12px 34px rgba(0,0,0,.55);
}
#tour-tip .tt-h { color: #5eead4; font-weight: 800; font-size: 12px; letter-spacing: .6px;
    text-transform: uppercase; margin-bottom: 5px; }
#tour-tip .tt-b { font-size: 14px; line-height: 1.5; color: #e6edf7; }
#tour-tip .tt-f { margin-top: 12px; display: flex; justify-content: space-between; align-items: center; }
#tour-tip .tt-f > span:first-child { color: #9fb0cc; font-size: 12px; }
#tour-tip .tt-btns { display: flex; gap: 8px; }
#tour-tip #tour-skip { background: transparent; border: 1px solid #24304d; color: #b7c4dc;
    border-radius: 8px; padding: 5px 12px; font-size: 12px; cursor: pointer; }
#tour-tip #tour-skip:hover { border-color: var(--teal); color: #ffffff; }
#tour-tip .tour-primary { background: linear-gradient(135deg, var(--teal), var(--teal-2)); border: none;
    color: #062522; border-radius: 8px; padding: 5px 14px; font-size: 12px; font-weight: 800; cursor: pointer; }
#tour-tip .tour-primary:hover { filter: brightness(1.08); }
/* ===== hero banner — centred, gradient title ===== */
#hero {
    position: relative;
    background: linear-gradient(135deg, #0b3b4a 0%, #0f766e 48%, #134e5e 100%);
    border: 1px solid #1c6f6a;
    border-radius: 20px;
    padding: 36px 34px 34px 34px;
    margin-bottom: 20px;
    text-align: center;
    box-shadow: 0 16px 48px rgba(3, 14, 30, .6), inset 0 0 60px rgba(45,212,191,.06);
    overflow: hidden;
}
#hero:before {
    content: ""; position: absolute; inset: 0;
    background: radial-gradient(600px 200px at 50% -30%, rgba(103,232,249,.28), transparent 70%);
    pointer-events: none;
}
#hero h1 {
    margin: 0 0 10px 0 !important; font-size: 40px !important; font-weight: 800 !important;
    letter-spacing: .4px; line-height: 1.1;
    background: linear-gradient(90deg, #ffffff 0%, #a7f3d0 45%, #67e8f9 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent !important;
    text-shadow: 0 2px 24px rgba(103,232,249,.25);
}
#hero p  { margin: 0 auto !important; color: #d7fbf5 !important; font-size: 16.5px; max-width: 780px; }
#hero .pill {
    display: inline-block; margin-top: 16px; padding: 7px 17px; border-radius: 999px;
    background: rgba(255,255,255,.14); color: #ecfeff; font-size: 13.5px; font-weight: 800;
    border: 1px solid rgba(103,232,249,.5); box-shadow: 0 0 18px rgba(45,212,191,.25);
}
/* ===== tabs — force clearly visible labels ===== */
.tabs, .tab-nav { background: transparent !important; border-bottom: 1px solid var(--stroke) !important; }
.tabs button, .tab-nav button, button[role="tab"], [role="tablist"] button {
    color: #eaf1fb !important; -webkit-text-fill-color: #eaf1fb !important;
    background: rgba(22,33,59,.65) !important; opacity: 1 !important; filter: none !important;
    font-weight: 700 !important; font-size: 15px !important;
    border: 1px solid var(--stroke) !important; border-bottom: none !important;
    border-radius: 10px 10px 0 0 !important; margin-right: 4px !important; padding: 9px 18px !important;
}
.tabs button:hover, .tab-nav button:hover, button[role="tab"]:hover {
    color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
    background: rgba(45,212,191,.14) !important;
}
.tabs button.selected, .tab-nav button.selected, button[role="tab"][aria-selected="true"] {
    color: #0b1120 !important; -webkit-text-fill-color: #0b1120 !important;
    background: linear-gradient(135deg, var(--teal), var(--teal-2)) !important;
    border-color: var(--teal) !important; font-weight: 800 !important;
}
/* ===== cards / panels ===== */
.panel-card {
    background: linear-gradient(180deg, var(--bg-2) 0%, var(--bg-1) 100%) !important;
    border: 1px solid var(--stroke) !important;
    border-radius: 16px !important;
    padding: 20px 22px !important;
    box-shadow: 0 6px 22px rgba(3, 10, 24, .45) !important;
}
.panel-card, .panel-card * { color: var(--text); }
.section-title { color: var(--teal) !important; font-weight: 800 !important; font-size: 13px !important;
    text-transform: uppercase; letter-spacing: 1.4px; margin: 2px 0 12px 2px !important; }
/* ===== component labels — visible on dark (fix light-on-light) ===== */
.gradio-container label, .gradio-container label span,
.panel-card label, .panel-card label span,
.block .label-wrap, .block .label-wrap span, span[data-testid="block-info"] {
    color: #e6edf7 !important; -webkit-text-fill-color: #e6edf7 !important;
    background: transparent !important; font-weight: 600 !important; opacity: 1 !important;
}
.label-wrap, .head, .block > .label-wrap { background: transparent !important; }
/* ===== image upload toolbar (Upload / Webcam / Paste) — keep visible & aligned ===== */
#in-image .source-selection, .image-container .source-selection, .image-frame .source-selection {
    display: flex !important; gap: 8px !important; justify-content: center !important;
    align-items: center !important; flex-wrap: wrap !important; padding: 8px !important;
    background: transparent !important;
}
/* style ONLY the source-selection toolbar buttons — never the preview wrapper */
#in-image .source-selection button, .image-container .source-selection button,
.image-frame .source-selection button, .upload-container .source-selection button {
    background: #1a2740 !important; border: 1px solid var(--stroke) !important; border-radius: 8px !important;
    color: #dbe4f3 !important; -webkit-text-fill-color: #dbe4f3 !important; opacity: 1 !important;
    padding: 6px 10px !important; min-width: 34px !important; display: inline-flex !important;
    align-items: center !important; justify-content: center !important;
}
#in-image .source-selection button:hover, .upload-container .source-selection button:hover { border-color: var(--teal) !important; }
#in-image .source-selection svg, .upload-container .source-selection svg {
    color: #dbe4f3 !important; stroke: #dbe4f3 !important; opacity: 1 !important;
    width: 20px !important; height: 20px !important;
}
#in-image .wrap, #in-image .upload-text, .upload-container .or, .wrap.default {
    color: #b7c4dc !important; -webkit-text-fill-color: #b7c4dc !important;
}
/* make sure the uploaded preview image actually shows (it is wrapped in a button) */
#in-image img, #in-image .image-frame img, .image-container img {
    display: block !important; opacity: 1 !important;
    max-height: 320px !important; width: auto !important; max-width: 100% !important;
    object-fit: contain !important; margin: 0 auto !important; min-width: 0 !important;
}
#in-image .image-frame, #in-image .image-container, #in-image button.image-button {
    min-width: 0 !important; width: 100% !important; display: block !important;
}
/* ===== example chips — neatly aligned, wrapped row, BLACK text on light-cyan ===== */
.examples { padding-top: 4px !important; }
.examples .table-wrap, .examples table, .examples tbody, [class*="samples"] {
    display: flex !important; flex-wrap: wrap !important; gap: 8px !important;
    border: none !important; background: transparent !important; width: 100% !important;
}
.examples thead { display: none !important; }
.examples tr { display: contents !important; }
.examples td, .examples button, .gr-sample-textbox, [class*="sample"] {
    display: inline-flex !important; align-items: center !important; white-space: normal !important;
    text-align: left !important; margin: 0 !important; padding: 7px 12px !important; line-height: 1.35 !important;
    background: #cffafe !important; color: #041014 !important; -webkit-text-fill-color: #041014 !important;
    font-weight: 600 !important; border: 1px solid #67e8f9 !important; border-radius: 8px !important;
}
.examples td *, [class*="sample"] * { color: #041014 !important; -webkit-text-fill-color: #041014 !important; }
.examples td:hover, .examples button:hover, [class*="sample"]:hover {
    background: #ecfeff !important; color: #000000 !important; border-color: #22d3ee !important;
}
/* ===== inputs ===== */
textarea, input[type=text] {
    background: var(--bg-0) !important; color: #ffffff !important;
    border: 1px solid var(--stroke) !important; border-radius: 10px !important;
}
textarea::placeholder, input[type=text]::placeholder { color: #7d8db0 !important; }
/* answer button */
#answer-btn {
    background: linear-gradient(135deg, var(--teal) 0%, var(--teal-2) 100%) !important;
    color: #062522 !important; -webkit-text-fill-color: #062522 !important;
    font-weight: 800 !important; border: none !important; border-radius: 10px !important;
    box-shadow: 0 6px 18px rgba(45,212,191,.28) !important;
}
#answer-btn:hover { filter: brightness(1.07); }
/* answer output */
#answer-out textarea {
    font-size: 18px !important; font-weight: 700 !important; color: #f0fdfa !important;
    background: var(--bg-0) !important; border: 1px solid var(--teal-2) !important;
}
/* ===== styled sections (About / Results) ===== */
.sec-h { color: #f0fdfa !important; font-size: 24px; margin: 0 0 10px 0; }
.sec-lead { color: #e6edf7; font-size: 15.5px; line-height: 1.6; max-width: 940px; margin: 0 0 8px 0; }
.sec-sub  { color: var(--muted); font-size: 14px; margin: 12px 0 10px 0; }
.sec-foot { color: #8ea0bf; font-size: 12.5px; margin-top: 18px; font-style: italic; }
.card-grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 6px; }
.mini { background: var(--bg-2); border: 1px solid var(--stroke); border-radius: 14px; padding: 16px; }
.mini p { color: #dbe4f3; font-size: 13.5px; line-height: 1.55; margin: 8px 0 0 0; }
.tag { display:inline-block; padding: 3px 11px; border-radius: 999px; font-size: 12px; font-weight: 800; letter-spacing:.4px; }
.tag-red   { background: rgba(248,113,113,.18); color: #fca5a5; border: 1px solid rgba(248,113,113,.4); }
.tag-green { background: rgba(74,222,128,.16);  color: #86efac; border: 1px solid rgba(74,222,128,.4); }
.tag-teal  { background: rgba(45,212,191,.16);  color: #5eead4; border: 1px solid rgba(45,212,191,.45); }
.res-table { border-collapse: collapse; width: 100%; margin: 6px 0 14px 0; font-size: 14.5px; }
.res-table th { background: var(--bg-2); color: var(--teal); text-align: left; padding: 10px 14px; border: 1px solid var(--stroke); }
.res-table td { color: #e6edf7; padding: 10px 14px; border: 1px solid var(--stroke); }
.res-table tr.best td { background: rgba(45,212,191,.08); }
.res-table .chip { background: rgba(56,189,248,.18); color: #7dd3fc; font-size: 11px; padding: 2px 8px; border-radius: 999px; margin-left: 6px; }
.chart-wrap { background: var(--bg-0); border: 1px solid var(--stroke); border-radius: 14px; padding: 14px 16px; }
.callout { margin-top: 14px; background: rgba(45,212,191,.08); border-left: 4px solid var(--teal);
    border-radius: 10px; padding: 12px 16px; color: #dbe4f3; font-size: 13.5px; line-height: 1.55; }
/* images */
.image-container, .image-frame { border-radius: 12px !important; border: 1px solid var(--stroke) !important; }
footer { display: none !important; }
"""

HERO = """
<div id="hero">
  <h1>🛰️ Remote-Sensing Visual Question Answering</h1>
  <p>Locally fine-tuned <b>Qwen2.5-VL-7B</b> · 4-bit · answering questions about satellite &amp; aerial imagery.</p>
  <span class="pill">★ Best config · 56.7% exact-match · 66.7% LLM-judge</span>
</div>
"""


def build(args):
    if args.mock:
        def run(image, question):
            return "Example answer", image
        return run

    import torch
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2_5_VLForConditionalGeneration,
                              AutoModelForZeroShotObjectDetection)
    from qwen_vl_utils import process_vision_info
    from peft import PeftModel

    # ---- VLM (7B fine-tuned, 4-bit on GPU) ----
    kw = dict(dtype="auto", device_map={"": 0})
    if args.load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    t0 = time.perf_counter()
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, **kw)
    model = PeftModel.from_pretrained(base, adapter_7b())
    proc = AutoProcessor.from_pretrained(MODEL, max_pixels=512 * 28 * 28)
    print(f"VLM ready in {time.perf_counter()-t0:.0f}s (7B fine-tuned)")

    # ---- detector (Grounding DINO) on CPU, loaded lazily on first box request ----
    gd = {}
    def load_detector():
        if "model" in gd:
            return
        gid = "IDEA-Research/grounding-dino-tiny"
        gd["proc"] = AutoProcessor.from_pretrained(gid)
        gd["model"] = AutoModelForZeroShotObjectDetection.from_pretrained(gid).to("cpu")
        print("Detector ready (CPU).")

    def answer(image, question):
        if image is None or not str(question).strip():
            return "⚠️ Upload an image and type a question."
        image.convert("RGB").save(TMP)
        messages = [{"role": "system", "content": DEMO_SYS},
                    {"role": "user", "content": [
                        {"type": "image", "image": str(TMP)},
                        {"type": "text", "text": str(question).strip()}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(messages)
        inputs = proc(text=[text], images=imgs, videos=vids,
                      padding=True, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=128)
        seq = out[:, inputs.input_ids.shape[1]:]
        ans = proc.batch_decode(seq, skip_special_tokens=True)[0]
        ans = " ".join(ans.split()).strip()
        return (ans[0].upper() + ans[1:]) if ans else "No answer produced."

    def boxes(image):
        if image is None:
            return None
        try:
            load_detector()
            from PIL import ImageDraw
            pil = image.convert("RGB")
            inputs = gd["proc"](images=pil, text=RS_CLASSES, return_tensors="pt")
            with torch.no_grad():
                outputs = gd["model"](**inputs)
            tgt = [pil.size[::-1]]          # (height, width)
            # transformers changed this API across versions — try each variant.
            res = None
            attempts = [
                lambda: gd["proc"].post_process_grounded_object_detection(
                    outputs, threshold=0.30, text_threshold=0.25, target_sizes=tgt),
                lambda: gd["proc"].post_process_grounded_object_detection(
                    outputs, inputs.input_ids, box_threshold=0.30,
                    text_threshold=0.25, target_sizes=tgt),
                lambda: gd["proc"].post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=0.30,
                    text_threshold=0.25, target_sizes=tgt),
            ]
            for fn in attempts:
                try:
                    res = fn()[0]; break
                except TypeError:
                    continue
            if res is None:
                print("box post-process failed for all API variants", flush=True)
                return pil
            labels = res.get("text_labels") or res.get("labels") or [""] * len(res["boxes"])
            img = pil.copy(); d = ImageDraw.Draw(img)
            for box, score, lab in zip(res["boxes"], res["scores"], labels):
                x1, y1, x2, y2 = [float(v) for v in box]
                d.rectangle([x1, y1, x2, y2], outline="red", width=3)
                d.text((x1 + 3, max(0, y1 - 12)), f"{lab} {float(score):.2f}", fill="red")
            return img
        except Exception as e:
            print("boxes() error:", e, flush=True)
            return image                    # return original image so the UI never crashes

    def run(image, question):
        """Single action: produce the answer AND always draw the boxes."""
        if image is None or not str(question).strip():
            return "⚠️ Upload an image and type a question.", None
        return answer(image, question), boxes(image)

    return run


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--share", action="store_true")
    return ap.parse_args(["--load-4bit"] if "ipykernel" in sys.modules else None)


def main():
    args = parse_args()
    import gradio as gr
    run = build(args)
    chart = bar_chart_svg()

    theme = gr.themes.Soft(primary_hue="teal", neutral_hue="slate").set(
        body_background_fill="#0b1120",
        block_background_fill="#111a2e",
        block_border_color="#24304d",
        input_background_fill="#0b1120",
    )
    with gr.Blocks(theme=theme, css=CSS, js=TOUR_JS, analytics_enabled=False,
                   title="FM-12a — Remote-Sensing Visual QA") as demo:
        demo.load(None, None, None, js=TOUR_JS)   # reliable on-load trigger for the coach tour
        gr.HTML(HERO + ("<p style='color:#fca5a5;text-align:center;margin:6px 2px'>⚙️ MOCK MODE — UI only</p>"
                        if args.mock else ""))
        with gr.Tabs():
            with gr.Tab("🛰️  Demo"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, elem_classes="panel-card"):
                        gr.Markdown("<div class='section-title'>Input</div>")
                        img = gr.Image(type="pil", label="Satellite / aerial image",
                                       height=320, elem_id="in-image")
                        q = gr.Textbox(label="Question", lines=2, elem_id="in-question",
                                       placeholder="e.g. How many ships are in the image?")
                        gr.Examples(EXAMPLES, inputs=q, label="Try an example (click to use)")
                        ans_btn = gr.Button("⚡  Answer", variant="primary", elem_id="answer-btn")
                    with gr.Column(scale=5, elem_classes="panel-card"):
                        gr.Markdown("<div class='section-title'>Answer</div>")
                        ans_box = gr.Textbox(label="Model answer", lines=2, elem_id="answer-out")
                        gr.Markdown("<div class='section-title' style='margin-top:12px'>Detected objects</div>")
                        boxed = gr.Image(label="Bounding boxes", height=320)
                ans_btn.click(run, [img, q], [ans_box, boxed])
                q.submit(run, [img, q], [ans_box, boxed])
            with gr.Tab("ℹ️  About"):
                with gr.Column(elem_classes="panel-card"):
                    gr.HTML(ABOUT_HTML)
            with gr.Tab("📊  Results"):
                with gr.Column(elem_classes="panel-card"):
                    gr.HTML(results_html(chart))

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()

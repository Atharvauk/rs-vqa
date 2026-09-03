"""
demo_UI.py — FM-12a Remote-Sensing Visual QA: a friendly, self-explaining web UI (7B).

Keeps all existing features (image+question -> answer, BEFORE vs AFTER fine-tuning),
and adds an About / Results / Guide so a first-time user understands the project and
can walk themselves through the demo. Lightweight: local only, built-in theme,
no external images, analytics disabled, ONE model in memory (fits 8 GB).

Run:
    python demo_UI.py --load-4bit --adapter qlora_adapter_v2
    # public link: add --share   |   UI-only test (no model): --mock
Open http://127.0.0.1:7860
"""
import argparse, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TMP  = HERE / "_demo_tmp.png"
DEMO_SYS = ("You are answering a question about an overhead satellite or aerial image. "
            "Give one clear, complete answer as a short word or phrase. Do not explain.")

NOTEBOOK_ARGS = ["--load-4bit", "--adapter", "qlora_adapter_v2"]

# ---------------------------------------------------------------- text content
ABOUT_MD = """
## 🛰️ About this project

This is a **Remote-Sensing Visual Question Answering (RS-VQA)** system: you give it a
**satellite / aerial image** and ask a question about the objects in it, and it answers
in a short phrase.

- **Model:** Qwen2.5-VL-7B, **fine-tuned** on the VRSBench benchmark with QLoRA, running
  **locally in 4-bit** (no paid cloud APIs in the answering path).
- **Focus:** *reliability*, not just raw accuracy — in remote sensing, a confident wrong
  answer is worse than an honest "not sure".
- **What was studied:**
  - **RQ1** — does adding an object detector's evidence help? *It actually hurts* (the
    detector is trained on ordinary photos, not overhead imagery).
  - **RQ2** — a rule-based **consistency check** that abstains on likely hallucinations
    improves reliability.
  - **Fine-tuning** — a carefully configured QLoRA fine-tune (answer-only loss, low learning
    rate) lifts accuracy *without* breaking hard question types like counting.

*Module ECS8060 — Artificial Intelligence Engineering · Queen's University Belfast ·
Atharva Uday Kalase (40509608).*
"""

RESULTS_MD = """
## 📊 Results & how it works

**How it answers:** the image and your question go straight to the vision-language model,
which reads the image (as visual tokens) and generates a short answer.

**The "Before vs After" toggle** shows the *same* model with the fine-tuning adapter
switched **off** (base model) vs **on** (fine-tuned) — so you can see exactly what the
domain fine-tuning changed.

**Headline accuracy** (on a fixed 1,000-question VRSBench test set — exact-match / lenient LLM-judge):

| Model | Zero-shot | Fine-tuned |
|---|---|---|
| Qwen2.5-VL 3B | 51.0% / 60.8% | 51.5% / 61.3% |
| **Qwen2.5-VL 7B (this demo)** | 51.3% / 62.6% | **56.7% / 66.7%  ← best** |
| Qwen3.6-35B (general, API) | 42.8% / 53.5% | — |

**Key takeaway:** naive additions (off-the-shelf grounding) fail in this domain, while
targeted ones (a consistency check, answer-only fine-tuning) succeed — and a small
*specialised* model beats a much larger *general* one. *Domain fit beats scale.*
"""

GUIDE_MD = """
## ❓ How to use this demo (30-second guide)

1. Open the **🛰️ Demo** tab.
2. **Upload** a satellite / aerial image (top-down view works best).
3. **Type a targeted question** — short and specific, not "describe the image".
4. Pick the **model**: *After fine-tuning* (best), *Before fine-tuning*, or *Compare both*.
5. Click **Answer**.

**Good example questions** (the model is strongest on these):
- "Is there a swimming pool in the image?"
- "What colour is the large building?"
- "Is this an urban or a rural area?"
- "What is the large object in the centre?"
- "How many ships are in the harbour?"

**Tip:** the model is strong on *existence*, *colour* and *scene* questions, and weaker on
exact *counting* — that is expected and is part of the research findings.
"""

EXAMPLE_QUESTIONS = [
    "Is there a swimming pool in the image?",
    "What colour is the large building?",
    "Is this an urban or a rural area?",
    "What is the large object in the centre?",
    "How many ships are in the harbour?",
    "What type of scene is shown?",
]


# ---------------------------------------------------------------- model
def build(args):
    if args.mock:
        def infer(image_path, question, use_adapter):
            return ("Fine-tuned: example answer" if use_adapter else "Base: example answer")
        return infer

    import torch
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2_5_VLForConditionalGeneration)
    from qwen_vl_utils import process_vision_info
    from peft import PeftModel

    kw = dict(dtype="auto", device_map={"": 0})
    if args.load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    t0 = time.perf_counter()
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, **kw)
    model = PeftModel.from_pretrained(base, args.adapter)     # adapter attached + enabled
    proc = AutoProcessor.from_pretrained(args.model, max_pixels=512 * 28 * 28)
    print(f"Model ready in {time.perf_counter()-t0:.0f}s (base + adapter)")

    def gen(image_path, question):
        messages = [{"role": "system", "content": DEMO_SYS},
                    {"role": "user", "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": question}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(messages)
        inputs = proc(text=[text], images=imgs, videos=vids,
                      padding=True, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=128)
        seq = out[:, inputs.input_ids.shape[1]:]
        ans = proc.batch_decode(seq, skip_special_tokens=True)[0]
        ans = " ".join(ans.split()).strip()
        return (ans[0].upper() + ans[1:]) if ans else "No answer produced."

    def infer(image_path, question, use_adapter):
        if use_adapter:
            return gen(image_path, question)
        with model.disable_adapter():            # base model = before fine-tuning
            return gen(image_path, question)
    return infer


# ---------------------------------------------------------------- args
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--adapter", default="qlora_adapter_v2")
    ap.add_argument("--mock", action="store_true", help="UI-only (no model)")
    ap.add_argument("--share", action="store_true", help="public link")
    in_notebook = "ipykernel" in sys.modules
    return ap.parse_args(NOTEBOOK_ARGS if in_notebook else None)


# ---------------------------------------------------------------- UI
def main():
    args = parse_args()
    import gradio as gr
    infer = build(args)

    def answer(image, question, choice):
        if image is None or not str(question).strip():
            return "⚠️ Please upload an image and type a question.", ""
        image.convert("RGB").save(TMP)
        q = str(question).strip()
        before = after = "—"
        if choice in ("Before fine-tuning", "Compare both"):
            before = infer(TMP, q, use_adapter=False)
        if choice in ("After fine-tuning", "Compare both"):
            after = infer(TMP, q, use_adapter=True)
        return before, after

    # analytics_enabled=False -> no telemetry pings (keeps it light + offline-friendly)
    with gr.Blocks(theme=gr.themes.Soft(), analytics_enabled=False,
                   title="FM-12a — Remote-Sensing Visual QA") as demo:
        gr.Markdown(
            "# 🛰️ Remote-Sensing Visual Question Answering\n"
            "Ask questions about objects in satellite / aerial images. "
            "Powered by a locally fine-tuned Qwen2.5-VL-7B model."
            + ("  \n*(MOCK MODE — no model loaded)*" if args.mock else ""))

        with gr.Tabs():
            # ---- DEMO TAB ----
            with gr.Tab("🛰️ Demo"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img = gr.Image(type="pil", label="1. Satellite / aerial image")
                        q = gr.Textbox(label="2. Your question",
                                       placeholder="e.g. How many ships are in the image?")
                        gr.Examples(EXAMPLE_QUESTIONS, inputs=q, label="Example questions (click to use)")
                        choice = gr.Radio(["After fine-tuning", "Before fine-tuning", "Compare both"],
                                          value="After fine-tuning", label="3. Model")
                        btn = gr.Button("Answer", variant="primary")
                    with gr.Column(scale=1):
                        after_box  = gr.Textbox(label="✅ After fine-tuning (best model)", lines=2)
                        before_box = gr.Textbox(label="Before fine-tuning (base model)", lines=2)
                        gr.Markdown("*Tip: use short, targeted questions. First answer may take "
                                    "a few seconds while the model warms up.*")
                btn.click(answer, [img, q, choice], [before_box, after_box])
                q.submit(answer, [img, q, choice], [before_box, after_box])

            # ---- ABOUT TAB ----
            with gr.Tab("ℹ️ About the project"):
                gr.Markdown(ABOUT_MD)

            # ---- RESULTS TAB ----
            with gr.Tab("📊 Results & how it works"):
                gr.Markdown(RESULTS_MD)

            # ---- GUIDE TAB ----
            with gr.Tab("❓ Guide"):
                gr.Markdown(GUIDE_MD)

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()

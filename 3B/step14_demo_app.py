"""
STEP 14 (7B) — Demonstration front-end (Gradio). CLEAN version.
Upload an aerial/satellite image, type a question, get ONE concise answer.
No technical read-outs (no confidence, boxes, evidence or timing) — just the answer.

Best answers come from the model directly (grounding is deliberately NOT injected here,
since the report shows it hurts accuracy). Use the fine-tuned adapter for the best output.

Run:
    pip install -U gradio transformers accelerate qwen-vl-utils pillow bitsandbytes
    python step14_demo_app.py --load-4bit
    # BEST OUTPUT (7B fine-tuned adapter):
    python step14_demo_app.py --load-4bit --adapter qlora_adapter_v2
    # UI-only test (no model):
    python step14_demo_app.py --mock
Open http://127.0.0.1:7860
"""
import argparse, sys, time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
TMP  = HERE / "_demo_tmp.png"

DEMO_SYS = ("You are answering a question about an overhead satellite or aerial image. "
            "Give one clear, complete answer as a short word or phrase. Do not explain.")

# In Jupyter a cell can't pass CLI flags — set them here (add the adapter for best output).
NOTEBOOK_ARGS = ["--load-4bit", "--adapter", "qlora_adapter_v2"]
# UI-only test:  NOTEBOOK_ARGS = ["--mock"]


def build_answer(args):
    if args.mock:
        def answer(image, question):
            if image is None or not str(question).strip():
                return "Upload an image and type a question."
            return "Example answer"
        return answer

    import step06_zeroshot_vqa as zs
    zargs = SimpleNamespace(model=args.model, load_4bit=args.load_4bit,
                            dry_run=False, adapter=args.adapter)
    t0 = time.perf_counter()
    predict = zs.make_predictor(zargs)
    zs.SYS = DEMO_SYS
    print(f"Model ready in {time.perf_counter()-t0:.0f}s")

    def answer(image, question):
        if image is None or not str(question).strip():
            return "Upload an image and type a question."
        image.convert("RGB").save(TMP)
        ans, _ = predict(TMP, str(question).strip())   # confidence intentionally ignored
        ans = " ".join(str(ans).split()).strip()
        if ans and ans[0].islower():
            ans = ans[0].upper() + ans[1:]
        return ans or "No answer produced."
    return answer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--adapter", default=None,
                    help="QLoRA adapter for best answers (e.g. qlora_adapter_v2)")
    ap.add_argument("--mock", action="store_true", help="UI demo without a model")
    ap.add_argument("--share", action="store_true", help="public Gradio link")
    in_notebook = "ipykernel" in sys.modules
    return ap.parse_args(NOTEBOOK_ARGS if in_notebook else None)


def main():
    args = parse_args()
    import gradio as gr
    answer = build_answer(args)

    with gr.Blocks(title="FM-12a — Remote-Sensing Visual QA") as demo:
        gr.Markdown("## Remote-Sensing Visual Question Answering\n"
                    "Upload a satellite / aerial image and ask a question about the objects in it.")
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="pil", label="Satellite / aerial image")
                q = gr.Textbox(label="Question",
                               placeholder="e.g. How many ships are in the image?")
                btn = gr.Button("Answer", variant="primary")
            with gr.Column(scale=1):
                out = gr.Textbox(label="Answer", lines=3)
        btn.click(answer, inputs=[img, q], outputs=out)
        q.submit(answer, inputs=[img, q], outputs=out)

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()

"""
demo_compare.py — Gradio demo that answers with the BASE model (before fine-tuning)
or the FINE-TUNED model (after), or BOTH side by side.

The base model is loaded once; the QLoRA adapter is toggled on/off (PEFT), so both
views come from ONE model in memory (~6.5 GB for 7B 4-bit — fits 8 GB).

RUN:
    pip install -U gradio transformers accelerate qwen-vl-utils pillow bitsandbytes peft
    python demo_compare.py                       # 7B + qlora_adapter_v2
    python demo_compare.py --model Qwen/Qwen2.5-VL-3B-Instruct --adapter qlora_adapter_v2
    # public link: add --share
Open http://127.0.0.1:7860
"""
import argparse, sys, time
from pathlib import Path

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
TMP = HERE / "_demo_tmp.png"

DEMO_SYS = ("You are answering a question about an overhead satellite or aerial image. "
            "Give one clear, complete answer as a short word or phrase. Do not explain.")


def tidy(a):
    a = " ".join(str(a).split()).strip()
    if a and a[0].islower():
        a = a[0].upper() + a[1:]
    return a or "No answer produced."


def build(args):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4")
    t0 = time.perf_counter()
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, quantization_config=bnb, device_map={"": 0})
    adapter_path = args.adapter if Path(args.adapter).is_absolute() else str(HERE / args.adapter)
    model = PeftModel.from_pretrained(base, adapter_path)
    proc = AutoProcessor.from_pretrained(args.model, max_pixels=512 * 28 * 28)
    model.eval()
    print(f"Loaded {args.model} + adapter '{args.adapter}' in {time.perf_counter()-t0:.0f}s")

    def gen(image_path, question):
        messages = [{"role": "system", "content": DEMO_SYS},
                    {"role": "user", "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": question}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(messages)
        inputs = proc(text=[text], images=imgs, videos=vids,
                      padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        ans = proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return tidy(ans)

    def answer_base(image_path, question):
        with model.disable_adapter():          # turn the adapter OFF -> original model
            return gen(image_path, question)

    def answer_ft(image_path, question):
        return gen(image_path, question)       # adapter ON -> fine-tuned model

    def respond(image, question, choice):
        if image is None or not str(question).strip():
            return "Upload an image and type a question.", ""
        image.convert("RGB").save(TMP)
        before = after = ""
        if choice in ("Before fine-tuning", "Both (compare)"):
            before = answer_base(TMP, question)
        if choice in ("After fine-tuning", "Both (compare)"):
            after = answer_ft(TMP, question)
        return before, after

    return respond


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default="qlora_adapter_v2")
    ap.add_argument("--share", action="store_true")
    in_notebook = "ipykernel" in sys.modules
    return ap.parse_args(["--model", "Qwen/Qwen2.5-VL-3B-Instruct",
                          "--adapter", "qlora_adapter_v2"] if in_notebook else None)


def main():
    args = parse_args()
    import gradio as gr
    respond = build(args)

    with gr.Blocks(title="FM-12a — Fine-tuning Before / After") as demo:
        gr.Markdown("## Remote-Sensing Visual QA — Before vs After Fine-tuning\n"
                    "Upload a satellite / aerial image, ask a question, and compare the "
                    "original model with the fine-tuned model.")
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="pil", label="Satellite / aerial image")
                q = gr.Textbox(label="Question", placeholder="e.g. What is the large object in the centre?")
                choice = gr.Radio(["Before fine-tuning", "After fine-tuning", "Both (compare)"],
                                  value="Both (compare)", label="Which model?")
                btn = gr.Button("Answer", variant="primary")
            with gr.Column(scale=1):
                out_before = gr.Textbox(label="Before fine-tuning (base model)", lines=3)
                out_after = gr.Textbox(label="After fine-tuning (adapter)", lines=3)
        btn.click(respond, inputs=[img, q, choice], outputs=[out_before, out_after])
        q.submit(respond, inputs=[img, q, choice], outputs=[out_before, out_after])

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()

"""
DEMO (7B) — Gradio front-end for FM-12a RS-VQA.
Upload an aerial image + question -> detected boxes, evidence, answer, confidence,
decision (answer/abstain/flag) and timing. Answering backend: Qwen2.5-VL-7B (local 4-bit).

RUN:
    pip install -U gradio transformers accelerate qwen-vl-utils pillow bitsandbytes openai
    python demo.py --load-4bit --adapter qlora_adapter_v2
    # UI-only test (no model):  python demo.py --mock
    # public link:              add --share
Open http://127.0.0.1:7860
"""
import argparse, json, random, sys, time
from pathlib import Path
from types import SimpleNamespace

try:
    HERE = Path(__file__).resolve().parent      # when run as a .py
except NameError:
    HERE = Path.cwd()                           # when pasted into a notebook cell
BASE = HERE
OUT  = BASE / "results"
OUT.mkdir(parents=True, exist_ok=True)

# Used ONLY inside Jupyter (a cell can't take CLI flags).
NOTEBOOK_ARGS = ["--load-4bit", "--adapter", "qlora_adapter_v2"]
# UI-only test:  NOTEBOOK_ARGS = ["--mock"]

GROUNDED_SYS = ("You are answering questions about an overhead aerial/satellite image. "
                "Use the detected-object list as evidence: check classes, counts and "
                "coordinates before answering. Answer with a short word or phrase only.")
BOX_THRESHOLD = 0.30


# ---------------------------------------------------------------- helpers
def draw_boxes(image, dets):
    from PIL import ImageDraw
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    for det in dets:
        x1, y1, x2, y2 = det["box_xyxy_px"]
        d.rectangle([x1, y1, x2, y2], outline="red", width=3)
        d.text((x1 + 3, max(0, y1 - 12)), f'{det["label"]} {det["score"]:.2f}', fill="red")
    return img


def to_box100(box_px, w, h):
    x1, y1, x2, y2 = box_px
    return [100 * x1 / w, 100 * y1 / h, 100 * x2 / w, 100 * y2 / h]


def evidence_text(dets, w, h, max_boxes=10):
    if not dets:
        return "No relevant objects were detected."
    dets = sorted(dets, key=lambda d: -d["score"])[:max_boxes]
    lines = [f"- {d['label']} (confidence {d['score']:.2f}) at "
             f"[{', '.join(f'{v:.0f}' for v in to_box100(d['box_xyxy_px'], w, h))}] "
             f"(0-100 scale, x1 y1 x2 y2)"
             for d in dets]
    return "Detected objects in the image:\n" + "\n".join(lines)


# ---------------------------------------------------------------- pipeline
def build_pipeline(args):
    from step09_calibration import apply_temperature
    from step07_grounding_detector import phrases_from_question
    import step10_consistency_check as cc

    T = 1.0
    cal = OUT / "calibration.json"
    if cal.exists():
        T = json.load(open(cal, encoding="utf-8"))["temperature"]
        print(f"Using temperature T={T:.2f} from results/calibration.json")
    else:
        print("No results/calibration.json found - using T=1.0 (uncalibrated)")

    if args.mock:
        print("MOCK MODE - no models loaded")
        rng = random.Random(0)

        def detect(img, phrases):
            return [{"label": phrases[0], "score": 0.87,
                     "box_xyxy_px": [img.width * .2, img.height * .2,
                                     img.width * .5, img.height * .5]}]

        def predict(img_path, q, t=None):
            return "2 (mock answer)", rng.uniform(0.4, 0.95)

        save_tmp = None
    else:
        from step07_grounding_detector import load_detector
        import step06_zeroshot_vqa as zs
        t0 = time.perf_counter()
        detect = load_detector("IDEA-Research/grounding-dino-tiny")
        zargs = SimpleNamespace(model=args.model, load_4bit=args.load_4bit,
                                dry_run=False, adapter=args.adapter)
        predict = zs.make_predictor(zargs)
        zs.SYS = GROUNDED_SYS
        save_tmp = BASE / "_demo_tmp.png"
        print(f"Models ready in {time.perf_counter()-t0:.0f}s")

    def answer(image, question, threshold):
        if image is None or not str(question).strip():
            return None, "Upload an image and type a question.", "", "", "", ""

        t_start = time.perf_counter()

        t0 = time.perf_counter()
        phrases = phrases_from_question(question)
        dets = [d for d in detect(image, phrases) if d["score"] >= BOX_THRESHOLD]
        t_detect = time.perf_counter() - t0

        ev = evidence_text(dets, image.width, image.height)
        boxed = draw_boxes(image, dets)

        t0 = time.perf_counter()
        if args.mock:
            raw_ans, raw_conf = predict(None, question)
        else:
            image.save(save_tmp)
            raw_ans, raw_conf = predict(save_tmp, f"{ev}\n\nQuestion: {question}")
        t_answer = time.perf_counter() - t0

        conf = apply_temperature(raw_conf, T)
        item = {"pred": raw_ans, "type": "demo", "question": question}
        dets10 = [{"label": d["label"], "score": d["score"],
                   "box_100": to_box100(d["box_xyxy_px"], image.width, image.height)}
                  for d in dets]
        consistent, reason = cc.check(item, dets10)

        if not consistent:
            badge = f"FLAGGED as possible hallucination - {reason}"
        elif conf < threshold:
            badge = f"ABSTAIN - calibrated confidence {conf:.2f} is below {threshold:.2f}"
        else:
            badge = f"ANSWERED - calibrated confidence {conf:.2f}"

        total = time.perf_counter() - t_start
        timing = (f"total {total:.1f}s  (detect {t_detect:.1f}s, "
                  f"answer {t_answer:.1f}s)   |   {len(dets)} boxes kept")
        conf_txt = f"{conf:.2f}   (raw {raw_conf:.2f}, T={T:.2f})"
        return boxed, raw_ans, conf_txt, badge, ev, timing

    return answer


# ---------------------------------------------------------------- args
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--adapter", default=None,
                    help="optional QLoRA adapter (e.g. qlora_adapter_v2)")
    ap.add_argument("--mock", action="store_true", help="UI demo without models")
    ap.add_argument("--share", action="store_true", help="public Gradio link")
    in_notebook = "ipykernel" in sys.modules
    return ap.parse_args(NOTEBOOK_ARGS if in_notebook else None)


# ---------------------------------------------------------------- main
def main():
    args = parse_args()
    import gradio as gr

    answer = build_pipeline(args)

    demo = gr.Interface(
        fn=answer,
        inputs=[gr.Image(type="pil", label="Aerial / satellite image"),
                gr.Textbox(label="Question",
                           placeholder="How many ships are in the image?"),
                gr.Slider(0.0, 1.0, value=0.5, label="Abstention threshold")],
        outputs=[gr.Image(label="Detected evidence (boxes)"),
                 gr.Textbox(label="Answer"),
                 gr.Textbox(label="Calibrated confidence"),
                 gr.Textbox(label="Decision"),
                 gr.Textbox(label="Evidence injected into the prompt", lines=6),
                 gr.Textbox(label="Timing")],
        title="FM-12a (7B) - Evidence-Verified RS-VQA",
        description=("Grounding DINO finds the evidence, Qwen2.5-VL-7B (local 4-bit) answers from it, "
                     "confidence decides whether to answer, and a consistency rule "
                     "flags answers that contradict the evidence."
                     + (" - MOCK MODE (fake model outputs)" if args.mock else "")))

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()

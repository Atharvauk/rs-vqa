"""
STEP 6 — Zero-shot VQA baseline (Qwen2.5-VL) with per-answer confidence.
Saves: results/preds_zeroshot.json   (used by steps 8, 9, 10, 12)

Live output — one line per question:
  [  12/1000]  13.2s | avg 12.8s/q | eta 210.6m | acc 64.2% | conf 0.72 | quantity | gt='2' pred='2' OK

Features:
  * auto-selects GPU 4-bit / GPU fp16 / CPU based on the VRAM actually found
  * per-iteration time, running average, ETA and running accuracy
  * works from a terminal (CLI flags) AND inside Jupyter (NOTEBOOK_ARGS)

QUICK TEST (no GPU, no model — verifies harness, produces fake preds):
    python step06_zeroshot_vqa.py --dry-run --limit 200        (~15 seconds)

REAL RUN:
    pip install -U transformers accelerate qwen-vl-utils pillow bitsandbytes
    python step06_zeroshot_vqa.py --limit 200 --load-4bit
    3B needs ~4.6 GB VRAM for 4-bit; on less it falls back to CPU (slow).
    For a 4 GB laptop GPU use:  --model Qwen/Qwen2-VL-2B-Instruct
"""
import argparse, json, collections, random, sys, time
from pathlib import Path
from step05_eval_harness import score_items, print_scores

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "VRSBench_data"
VQA  = DATA / "VRSBench_EVAL_vqa.json"
IMG  = DATA / "Images_val" / "Images_val"
OUT  = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

# Used ONLY when running inside Jupyter (a notebook cell can't take CLI flags).
NOTEBOOK_ARGS = ["--limit", "1000", "--load-4bit", "--out", "preds_zeroshot.json"]  # 7B zero-shot
# dry test:   NOTEBOOK_ARGS = ["--dry-run", "--limit", "200"]

SYS = ("You are answering questions about an overhead aerial/satellite image. "
       "Answer with a short word or phrase only, no explanation.")


# ---------------------------------------------------------------- helpers
def load_items(limit, seed=0):
    """Fixed random subset of the VQA eval set (same seed => same questions)."""
    vqa = json.load(open(VQA, encoding="utf-8"))
    random.Random(seed).shuffle(vqa)
    return vqa[:limit] if limit else vqa


def quick_correct(pred, gt):
    """Loose match, for the LIVE display only. score_items() is authoritative."""
    p = str(pred).strip().lower().rstrip(".")
    g = str(gt).strip().lower().rstrip(".")
    return p == g or (g and g in p) or (p and p in g and len(p) > 1)


def fmt_eta(seconds):
    """Readable ETA: seconds -> '3.2m' or '1.4h'."""
    if seconds < 90:
        return f"{seconds:4.0f}s"
    if seconds < 5400:
        return f"{seconds/60:5.1f}m"
    return f"{seconds/3600:5.1f}h"


# ---------------------------------------------------------------- model
def make_predictor(args):
    """Returns predict(image_path, question, qtype) -> (answer, confidence 0-1)."""
    if args.dry_run:
        vqa_all = json.load(open(VQA, encoding="utf-8"))
        by_type = collections.defaultdict(list)
        for x in vqa_all:
            by_type[x["type"]].append(str(x["ground_truth"]).strip().lower())
        majority = {t: collections.Counter(a).most_common(1)[0][0]
                    for t, a in by_type.items()}
        rng = random.Random(1)

        def predict(image_path, question, qtype=None):
            return majority.get(qtype, "yes"), rng.uniform(0.4, 0.95)
        return predict

    import torch
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    # right class for the chosen checkpoint:
    #   Qwen2.5-VL-3B/7B/72B -> Qwen2_5_VL...;  Qwen2-VL-2B/7B -> Qwen2VL...
    if "qwen2.5-vl" in args.model.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
    else:
        from transformers import Qwen2VLForConditionalGeneration as VLModel

    # --- pick a device/precision that actually fits this machine --------------
    use_cuda = torch.cuda.is_available()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if use_cuda else 0.0
    ml = args.model.lower()
    params_b = 7 if "7b" in ml else 3 if "3b" in ml else 2      # rough model size
    need_4bit = params_b * 0.7 + 2.5        # weights + vision encoder + activations
    need_fp16 = params_b * 2.2 + 2.5

    if use_cuda and args.load_4bit and vram_gb >= need_4bit:
        kw = dict(dtype="auto", device_map={"": 0})            # whole model on GPU
        mode = f"GPU 4-bit ({torch.cuda.get_device_name(0)}, {vram_gb:.1f} GB)"
    elif use_cuda and not args.load_4bit and vram_gb >= need_fp16:
        kw = dict(dtype="auto", device_map={"": 0})
        mode = f"GPU fp16 ({torch.cuda.get_device_name(0)}, {vram_gb:.1f} GB)"
    else:
        kw = dict(dtype="auto", device_map="cpu")              # safe fallback
        args.load_4bit = False                                 # bnb 4-bit needs CUDA
        why = ("no CUDA GPU" if not use_cuda
               else f"only {vram_gb:.1f} GB VRAM, need ~{need_4bit:.1f} GB")
        mode = f"CPU  (fallback: {why}) - SLOW"
    print(f"Loading {args.model}  ->  {mode}", flush=True)

    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    t_load = time.perf_counter()
    model = VLModel.from_pretrained(args.model, **kw)
    if getattr(args, "adapter", None):        # step13's QLoRA adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"(loaded fine-tuned adapter: {args.adapter})")
    proc = AutoProcessor.from_pretrained(args.model, max_pixels=512 * 28 * 28)
    print(f"Model ready in {time.perf_counter()-t_load:.0f}s", flush=True)

    def predict(image_path, question, qtype=None):
        messages = [
            {"role": "system", "content": SYS},
            {"role": "user", "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": question}]}]
        text = proc.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True)
        imgs, vids = process_vision_info(messages)
        inputs = proc(text=[text], images=imgs, videos=vids,
                      padding=True, return_tensors="pt").to(model.device)
        gen = model.generate(**inputs, max_new_tokens=128,
                             output_scores=True, return_dict_in_generate=True)
        seq = gen.sequences[:, inputs.input_ids.shape[1]:]
        answer = proc.batch_decode(seq, skip_special_tokens=True)[0]
        # confidence = mean probability of the generated tokens
        probs = [torch.softmax(step[0], dim=-1)[tok].item()
                 for step, tok in zip(gen.scores, seq[0])]
        conf = sum(probs) / len(probs) if probs else 0.0
        return answer, conf

    return predict


# ---------------------------------------------------------------- args
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--adapter", default=None,
                    help="path to a step13 QLoRA adapter (e.g. qlora_adapter)")
    ap.add_argument("--out", default="preds_zeroshot.json")
    # in Jupyter use NOTEBOOK_ARGS; from a terminal use the real CLI flags
    in_notebook = "ipykernel" in sys.modules
    return ap.parse_args(NOTEBOOK_ARGS if in_notebook else None)


# ---------------------------------------------------------------- main
def main():
    args = parse_args()
    items = load_items(args.limit)
    predict = make_predictor(args)

    n = len(items)
    width = len(str(n))
    records, running_correct = [], 0

    print(f"\n=== Zero-shot baseline: {n} questions ===")
    print(f"{'idx':>{2*width+3}} | {'time':>6} | {'avg':>8} | {'eta':>6} | "
          f"{'acc':>6} | {'conf':>5} | type / gt -> pred")
    print("-" * 100, flush=True)

    t_start = time.perf_counter()
    for i, x in enumerate(items, 1):
        t0 = time.perf_counter()
        ans, conf = predict(IMG / x["image_id"], x["question"], x["type"])
        dt = time.perf_counter() - t0

        ok = quick_correct(ans, x["ground_truth"])
        running_correct += ok
        records.append({"image_id": x["image_id"], "question": x["question"],
                        "type": x["type"], "gt": x["ground_truth"],
                        "pred": ans, "conf": round(conf, 4), "sec": round(dt, 2)})

        elapsed = time.perf_counter() - t_start
        avg = elapsed / i
        eta = avg * (n - i)
        print(f"[{i:>{width}}/{n}] | {dt:5.1f}s | {avg:6.1f}s/q | {fmt_eta(eta)} | "
              f"{100*running_correct/i:5.1f}% | {conf:5.2f} | "
              f"{x['type'][:16]:16} gt='{str(x['ground_truth'])[:14]}' "
              f"-> '{str(ans)[:14]}' {'OK ' if ok else 'X'}", flush=True)

    total = time.perf_counter() - t_start
    print("-" * 100)
    print(f"Finished {n} items in {total/60:.1f} min "
          f"({total/max(n,1):.1f} s/question average)")

    scores = score_items(records)          # adds 'correct' to each record
    print_scores(scores, label=("dry-run" if args.dry_run else args.model))

    json.dump({"meta": {"model": "dry-run" if args.dry_run else args.model,
                        "n": len(records), "condition": "zeroshot",
                        "total_sec": round(total, 1),
                        "avg_sec_per_q": round(total/max(len(records), 1), 2)},
               "scores": scores, "items": records},
              open(OUT / args.out, "w"), indent=1)
    print(f"\nSaved: {OUT / args.out}")


if __name__ == "__main__":
    main()

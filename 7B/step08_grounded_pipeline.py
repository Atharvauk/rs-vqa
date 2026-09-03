"""
STEP 8 — Grounded pipeline: detector boxes injected into the VLM prompt.
The RQ1 "treatment" condition. Needs results/detections.json from step 7
(in --dry-run it tolerates a missing file).
Saves: results/preds_grounded.json

QUICK TEST (no GPU):
    python step08_grounded_pipeline.py --dry-run --limit 200
    Time: ~15 seconds

REAL RUN (GPU; run step 7 first with the same --limit):
    python step08_grounded_pipeline.py --limit 200 --load-4bit
    Time (3B default): ~20-30 min/200 items on a T4; ~50-85 min on a
          4 GB laptop GPU. Longer prompts than step 6. 7B via --model:
          roughly double on a T4, DOES NOT FIT in 4 GB.
"""
import argparse, json
from pathlib import Path
from step05_eval_harness import score_items, print_scores
import step06_zeroshot_vqa as zs

OUT = Path(__file__).resolve().parent / "results"

def evidence_text(dets, max_boxes=10):
    if not dets:
        return "No relevant objects were detected."
    dets = sorted(dets, key=lambda d: -d["score"])[:max_boxes]
    lines = [f"- {d['label']} (confidence {d['score']:.2f}) at "
             f"[{', '.join(f'{v:.0f}' for v in d['box_100'])}] (0-100 scale, x1 y1 x2 y2)"
             for d in dets]
    return "Detected objects in the image:\n" + "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--adapter", default=None,
                    help="path to a step13 QLoRA adapter")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    det_path = OUT / "detections.json"
    detections = json.load(open(det_path)) if det_path.exists() else {}
    if not detections and not args.dry_run:
        raise SystemExit("results/detections.json missing — run step07 first (same --limit).")

    items = zs.load_items(args.limit)
    predict = zs.make_predictor(args)

    # grounded system prompt: reason from the evidence boxes
    zs.SYS = ("You are answering questions about an overhead aerial/satellite image. "
              "Use the detected-object list as evidence: check classes, counts and "
              "coordinates before answering. Answer with a short word or phrase only.")

    records = []
    for i, x in enumerate(items, 1):
        key = f"{x['image_id']}||{x['question']}"
        dets = detections.get(key, [])
        question = f"{evidence_text(dets)}\n\nQuestion: {x['question']}"
        ans, conf = predict(zs.IMG / x["image_id"], question, x["type"])
        records.append({"image_id": x["image_id"], "question": x["question"],
                        "type": x["type"], "gt": x["ground_truth"],
                        "pred": ans, "conf": round(conf, 4),
                        "n_detections": len(dets)})
        if i % 50 == 0:
            print(f"  {i}/{len(items)} done...", flush=True)

    scores = score_items(records)
    print_scores(scores, label="grounded pipeline")

    # side-by-side with the zero-shot run if available
    zs_path = OUT / "preds_zeroshot.json"
    if zs_path.exists():
        base = json.load(open(zs_path))["scores"]["overall_acc"]
        print(f"\nZero-shot overall: {100*base:.1f}%  ->  grounded overall: "
              f"{100*scores['overall_acc']:.1f}%  (delta {100*(scores['overall_acc']-base):+.1f} pts)")

    json.dump({"meta": {"model": "dry-run" if args.dry_run else args.model,
                        "n": len(records), "condition": "grounded"},
               "scores": scores, "items": records},
              open(OUT / "preds_grounded.json", "w"), indent=1)
    print(f"Saved: {OUT / 'preds_grounded.json'}")

if __name__ == "__main__":
    main()

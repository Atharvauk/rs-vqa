"""
STEP 7 — Grounding stage: open-vocabulary detector (Grounding DINO).
Detects question-relevant objects and saves boxes for the pipeline (step 8)
and consistency check (step 10). Also scores grounding Acc@IoU on the
referring set.
Saves: results/detections.json

Run (GPU recommended):
    pip install -U transformers accelerate pillow
    python step07_grounding_detector.py --limit 200
Time:  ~10-20 min for 200 images on a T4 with grounding-dino-tiny
       (~3-6 s/image); 3-4x slower on CPU; A100 ~3-5 min.
"""
import argparse, json, random, re
from pathlib import Path
from step03_coord_utils import parse_box_str, pixels_to_box100
from step05_eval_harness import acc_at_iou

DATA = Path(__file__).resolve().parent.parent / "VRSBench_data"
IMG  = DATA / "Images_val" / "Images_val"
OUT  = Path(__file__).resolve().parent / "results"

# common VRSBench/DOTA object classes, used to pull object words out of questions
OBJECT_VOCAB = ["plane", "airplane", "aircraft", "ship", "boat", "vehicle", "car",
                "truck", "bus", "helicopter", "bridge", "harbor", "building", "house",
                "storage tank", "tank", "roundabout", "swimming pool", "pool",
                "tennis court", "basketball court", "soccer field", "ground track field",
                "baseball diamond", "container", "crane", "road", "runway", "windmill"]

def phrases_from_question(q):
    ql = q.lower()
    found = [w for w in OBJECT_VOCAB if re.search(rf"\b{re.escape(w)}s?\b", ql)]
    return found or [q]          # fall back to the full phrase (referring queries)

def load_detector(model_id):
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc  = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    def detect(image, phrases, box_thr=0.30, text_thr=0.25):
        """image: PIL.Image -> [{'label','score','box_xyxy_px'}]"""
        text = ". ".join(phrases) + "."
        inputs = proc(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        res = proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=box_thr, text_threshold=text_thr,
            target_sizes=[image.size[::-1]])[0]
        return [{"label": l, "score": float(s), "box_xyxy_px": [float(v) for v in b]}
                for l, s, b in zip(res["labels"], res["scores"], res["boxes"])]
    return detect

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--score-referring", action="store_true",
                    help="also compute grounding Acc@IoU on the referring set")
    args = ap.parse_args()

    from PIL import Image
    OUT.mkdir(exist_ok=True)
    detect = load_detector(args.model)

    # 1) detections for the SAME VQA subset used in step 6 (same seed/limit)
    vqa = json.load(open(DATA / "VRSBench_EVAL_vqa.json", encoding="utf-8"))
    random.Random(0).shuffle(vqa)
    subset = vqa[:args.limit] if args.limit else vqa

    detections = {}
    for i, x in enumerate(subset, 1):
        key = f"{x['image_id']}||{x['question']}"
        p = IMG / x["image_id"]
        if not p.exists():
            detections[key] = []
            continue
        im = Image.open(p).convert("RGB")
        dets = detect(im, phrases_from_question(x["question"]))
        for d in dets:
            d["box_100"] = list(pixels_to_box100(d["box_xyxy_px"], *im.size))
        detections[key] = dets
        if i % 25 == 0:
            print(f"  {i}/{len(subset)} images done...", flush=True)

    json.dump(detections, open(OUT / "detections.json", "w"), indent=1)
    print(f"Saved: {OUT / 'detections.json'}  ({len(detections)} entries)")

    # 2) optional: grounding accuracy on referring expressions
    if args.score_referring:
        ref = json.load(open(DATA / "VRSBench_EVAL_referring.json", encoding="utf-8"))
        random.Random(0).shuffle(ref)
        ref = ref[:args.limit] if args.limit else ref
        pairs = []
        for i, x in enumerate(ref, 1):
            p = IMG / x["image_id"]
            if not p.exists():
                continue
            im = Image.open(p).convert("RGB")
            dets = detect(im, [x["question"]])
            best = max(dets, key=lambda d: d["score"], default=None)
            pred = pixels_to_box100(best["box_xyxy_px"], *im.size) if best else None
            pairs.append((pred, parse_box_str(x["ground_truth"])))
            if i % 25 == 0:
                print(f"  referring {i}/{len(ref)}...", flush=True)
        acc = acc_at_iou(pairs)
        print(f"Grounding Acc@0.5 = {100*acc[0.5]:.1f}%   Acc@0.7 = {100*acc[0.7]:.1f}%  (n={len(pairs)})")
        json.dump({"n": len(pairs), "acc@0.5": acc[0.5], "acc@0.7": acc[0.7],
                   "model": args.model},
                  open(OUT / "grounding_scores.json", "w"), indent=2)

if __name__ == "__main__":
    main()

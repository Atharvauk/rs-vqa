"""
STEP 3 — Coordinate utilities + IoU (self-tested).
VRSBench boxes come as '{<x1><y1><x2><y2>}' on a 0-100 scale.
Provides: parse_box_str, box100_to_pixels, pixels_to_box100, iou.
Run as a script it self-tests IoU and draws boxes on 5 sample images
(saved to results/box_check/).

Run:   python step03_coord_utils.py
Time:  ~10 seconds (CPU)
"""
import json, re, random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "VRSBench_data"
OUT  = Path(__file__).resolve().parent / "results" / "box_check"

def parse_box_str(s):
    """'{<25><40><33><60>}' -> (25.0, 40.0, 33.0, 60.0)  (x1,y1,x2,y2 in 0-100)"""
    nums = re.findall(r"<([\d.]+)>", str(s))
    if len(nums) != 4:
        raise ValueError(f"Bad box string: {s}")
    return tuple(float(n) for n in nums)

def box100_to_pixels(b, w, h):
    x1, y1, x2, y2 = b
    return (x1 / 100 * w, y1 / 100 * h, x2 / 100 * w, y2 / 100 * h)

def pixels_to_box100(b, w, h):
    x1, y1, x2, y2 = b
    return (x1 / w * 100, y1 / h * 100, x2 / w * 100, y2 / h * 100)

def iou(a, b):
    """Intersection-over-union of two xyxy boxes (implemented from scratch)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def _self_test():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9
    assert parse_box_str("{<25><40><33><60>}") == (25.0, 40.0, 33.0, 60.0)
    print("IoU + parser self-tests PASSED.")

def _draw_samples(n=5):
    from PIL import Image, ImageDraw
    OUT.mkdir(parents=True, exist_ok=True)
    ref = json.load(open(DATA / "VRSBench_EVAL_referring.json", encoding="utf-8"))
    img_dir = DATA / "Images_val" / "Images_val"
    picked = random.Random(0).sample(ref, n * 3)
    done = 0
    for x in picked:
        p = img_dir / x["image_id"]
        if not p.exists() or done >= n:
            continue
        im = Image.open(p).convert("RGB")
        box = box100_to_pixels(parse_box_str(x["ground_truth"]), *im.size)
        ImageDraw.Draw(im).rectangle(box, outline=(255, 0, 0), width=3)
        fn = OUT / f"{x['image_id'][:-4]}_gt.png"
        im.save(fn)
        print(f"  drew '{x['question'][:60]}...' -> {fn.name}")
        done += 1
    print(f"Saved {done} box-overlay images to {OUT}")

if __name__ == "__main__":
    _self_test()
    try:
        _draw_samples()
    except ImportError:
        print("(pillow not installed — skipped drawing)")

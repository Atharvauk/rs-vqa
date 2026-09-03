"""
STEP 1 — Setup check: verifies the VRSBench data is in place and reports
what compute is available. Run this first; every later step assumes it passes.

Run:   python step01_check_setup.py
Time:  ~1 second (CPU)
"""
import json, sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "VRSBench_data"

def check(label, ok, detail=""):
    mark = "OK " if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" — {detail}" if detail else ""))
    return ok

def main():
    all_ok = True
    print(f"Data folder: {DATA}\n")

    # 1. required files/folders
    required = {
        "VQA eval file":        DATA / "VRSBench_EVAL_vqa.json",
        "Referring eval file":  DATA / "VRSBench_EVAL_referring.json",
        "Caption eval file":    DATA / "VRSBench_EVAL_Cap.json",
        "Train file":           DATA / "VRSBench_train.json",
        "Val images folder":    DATA / "Images_val" / "Images_val",
        "Val annotations":      DATA / "Annotations_val",
    }
    for label, p in required.items():
        all_ok &= check(label, p.exists(), str(p.name))

    # 2. counts (only if the files are there)
    vqa_path = required["VQA eval file"]
    if vqa_path.exists():
        vqa = json.load(open(vqa_path, encoding="utf-8"))
        types = {}
        for x in vqa:
            types[x["type"]] = types.get(x["type"], 0) + 1
        check("VQA items loaded", len(vqa) > 0, f"{len(vqa)} questions, {len(types)} types")
        img_dir = required["Val images folder"]
        if img_dir.exists():
            n_img = sum(1 for _ in img_dir.glob("*.png"))
            check("Val images present", n_img > 0, f"{n_img} .png files")
            sample = vqa[0]["image_id"]
            check("Sample image resolvable", (img_dir / sample).exists(), sample)

    # 3. compute
    print()
    try:
        import torch
        gpu = torch.cuda.is_available()
        detail = torch.cuda.get_device_name(0) if gpu else "CPU only — steps 6-8 need Colab/Kaggle"
        check("PyTorch installed", True, f"v{torch.__version__}, GPU: {detail}")
    except ImportError:
        check("PyTorch installed", False, "fine for steps 1-5/9-12 dry-runs; needed for 6-8")
    for mod in ("PIL", "matplotlib"):
        try:
            __import__(mod)
            check(f"{mod} installed", True)
        except ImportError:
            all_ok &= check(f"{mod} installed", False, "pip install -r requirements.txt")

    print("\n" + ("Setup looks good — proceed to step02." if all_ok
                  else "Fix the FAIL lines above before continuing."))
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()

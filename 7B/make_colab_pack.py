"""
Build a small ZIP to upload to Colab for step-13 QLoRA fine-tuning.

Instead of shipping all 20,264 training images (several GB), this packs ONLY
the N examples you will actually train on, plus the training script.

Run:
    python make_colab_pack.py --train-n 500
Produces:
    colab_pack.zip   (~50-150 MB depending on --train-n)

Contents of the zip:
    rsvqa_pipeline/step13_qlora_finetune.py
    rsvqa_pipeline/VRSBench_data/VRSBench_train.json        (filtered to N items)
    rsvqa_pipeline/VRSBench_data/Images_train/Images_train/*.png
"""
import argparse, json, random, re, shutil, sys, tempfile, zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "VRSBench_data"
IMG  = DATA / "Images_train" / "Images_train"

NOTEBOOK_ARGS = ["--train-n", "500"]


def pick_items(n, seed=0):
    """Same selection logic as step13.load_train_items, but keeps raw records."""
    raw = json.load(open(DATA / "VRSBench_train.json", encoding="utf-8"))
    keep = []
    for r in raw:
        conv = r.get("conversations", [])
        if len(conv) < 2 or "[vqa]" not in conv[0]["value"]:
            continue
        if not (IMG / r["image"]).exists():
            continue
        keep.append(r)
    random.Random(seed).shuffle(keep)
    return keep[:n]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=500)
    ap.add_argument("--out", default="colab_pack.zip")
    in_notebook = "ipykernel" in sys.modules
    return ap.parse_args(NOTEBOOK_ARGS if in_notebook else None)


def main():
    args = parse_args()
    print(f"[1/4] Selecting {args.train_n} [vqa] training items...")
    records = pick_items(args.train_n)
    if not records:
        raise SystemExit("No usable training items found — is Images_train unpacked?")
    print(f"      got {len(records)} items")

    tmp = Path(tempfile.mkdtemp()) / "rsvqa_pipeline"
    (tmp / "VRSBench_data" / "Images_train" / "Images_train").mkdir(parents=True)

    print("[2/4] Copying images...")
    total_mb = 0
    for i, r in enumerate(records, 1):
        src = IMG / r["image"]
        dst = tmp / "VRSBench_data" / "Images_train" / "Images_train" / r["image"]
        shutil.copy2(src, dst)
        total_mb += src.stat().st_size / 1e6
        if i % 100 == 0:
            print(f"      {i}/{len(records)} ({total_mb:.0f} MB)", flush=True)
    print(f"      {len(records)} images, {total_mb:.0f} MB")

    print("[3/4] Writing filtered VRSBench_train.json + training script...")
    json.dump(records, open(tmp / "VRSBench_data" / "VRSBench_train.json", "w"), indent=1)
    shutil.copy2(BASE / "step13_qlora_finetune.py", tmp / "step13_qlora_finetune.py")

    print("[4/4] Zipping...")
    out_zip = BASE / args.out
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in tmp.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(tmp.parent))
    shutil.rmtree(tmp.parent, ignore_errors=True)

    print(f"\nDone: {out_zip}  ({out_zip.stat().st_size/1e6:.0f} MB)")
    print("Upload this file to Colab (see the Colab cells your assistant gave you).")


if __name__ == "__main__":
    main()

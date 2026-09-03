"""
download_vrsbench.py — download the FULL VRSBench dataset into THIS folder.

Everything lands under  <this folder>/VRSBench_data/  which is exactly where the
7B step files here look for it. Run this once on the Alienware before the pipeline.

    pip install -U huggingface_hub
    python download_vrsbench.py

Size: ~13 GB (29,614 images). 10-40 min depending on connection.
Resumable: if it stops, just run it again — it continues where it left off.
"""
from pathlib import Path
import zipfile, shutil, sys

HERE = Path(__file__).resolve().parent
DEST = HERE / "VRSBench_data"
DEST.mkdir(parents=True, exist_ok=True)
REPO = "xiang709/VRSBench"

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("Missing dependency. Run:  pip install -U huggingface_hub   then re-run.")

print(f"Downloading {REPO}\n  -> {DEST}\n(~13 GB, resumable)\n")
snapshot_download(
    repo_id=REPO, repo_type="dataset", local_dir=str(DEST),
    resume_download=True,
    allow_patterns=["*.json", "Images_*.zip", "Annotations_*.zip"],
)

# ---- unzip the image archives into split subfolders ----
def unzip_images(split):
    z = DEST / f"Images_{split}.zip"
    if not z.exists():
        print(f"  (no {z.name}, skipping)"); return
    out = DEST / f"Images_{split}"
    out.mkdir(exist_ok=True)
    print(f"Unzipping {z.name} -> {out.name}\\ ...")
    with zipfile.ZipFile(z) as zf:
        zf.extractall(out)

for split in ("train", "val"):
    unzip_images(split)

# ---- normalise nesting so images sit at Images_<split>/Images_<split>/*.png ----
def ensure_nested(split):
    base   = DEST / f"Images_{split}"
    nested = base / f"Images_{split}"
    if nested.is_dir() and any(nested.glob("*.png")):
        return f"Images_{split}: OK (nested)"
    pngs = list(base.glob("*.png"))
    if pngs:                                   # images one level too high -> push down
        nested.mkdir(parents=True, exist_ok=True)
        for p in pngs:
            p.rename(nested / p.name)
        return f"Images_{split}: fixed nesting ({len(pngs)} images)"
    for d in base.rglob("*"):                  # extracted under some other subdir
        if d.is_dir() and any(d.glob("*.png")):
            if d != nested:
                if nested.exists(): shutil.rmtree(nested)
                d.rename(nested)
            return f"Images_{split}: located + nested"
    return f"Images_{split}: NO PNGs FOUND - check manually"

for split in ("train", "val"):
    print(" ", ensure_nested(split))

# ---- verification ----
print("\n--- verification (all should say OK) ---")
json_checks = {"VRSBench_EVAL_vqa.json": DEST / "VRSBench_EVAL_vqa.json",
               "VRSBench_train.json":    DEST / "VRSBench_train.json"}
for name, p in json_checks.items():
    print(("OK  " if p.exists() else "MISSING "), name)
for split in ("val", "train"):
    d = DEST / f"Images_{split}" / f"Images_{split}"
    n = len(list(d.glob("*.png"))) if d.is_dir() else 0
    print((f"OK  {n} images" if n else "MISSING "), f"Images_{split}/Images_{split}")
print("\nDone. If anything says MISSING, paste this output and I'll adjust.")

"""
unzip_vrsbench.py — extract the VRSBench zips that are already downloaded.

Run it from inside pipeline_7b (where VRSBench_data/ lives):
    python unzip_vrsbench.py

Extracts:
  Images_train.zip / Images_val.zip  -> VRSBench_data/Images_<split>/Images_<split>/*.png
  Annotations_train.zip / Annotations_val.zip -> VRSBench_data/Annotations_<split>/
Uses the OS `tar` (fast) if present, else parallel Python.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import zipfile, shutil, subprocess, time, sys

# --- location of the zips. Edit this line if your VRSBench_data is elsewhere. ---
DEST = Path(__file__).resolve().parent / "VRSBench_data"
if not DEST.exists():
    DEST = Path.cwd() / "VRSBench_data"          # fallback: current folder
print("Using data folder:", DEST)
if not DEST.exists():
    sys.exit("VRSBench_data not found. cd into pipeline_7b, or edit DEST above.")


def extract(zip_path, out_dir):
    """Fast extract: OS tar first, else parallel zipfile (thread-safe handles)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if shutil.which("tar"):
        rc = subprocess.run(["tar", "-xf", str(zip_path), "-C", str(out_dir)]).returncode
        if rc == 0:
            print(f"   {zip_path.name}: done via tar in {time.perf_counter()-t0:.0f}s")
            return
        print("   tar failed; using parallel python ...")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    def worker(chunk):
        with zipfile.ZipFile(zip_path) as zf:
            for n in chunk:
                zf.extract(n, out_dir)
    k = 8
    chunks = [names[i::k] for i in range(k)]
    with ThreadPoolExecutor(max_workers=k) as ex:
        list(ex.map(worker, chunks))
    print(f"   {zip_path.name}: done via parallel python in {time.perf_counter()-t0:.0f}s")


def ensure_nested(split):
    """Guarantee images sit at Images_<split>/Images_<split>/*.png (what step06 wants)."""
    base   = DEST / f"Images_{split}"
    nested = base / f"Images_{split}"
    if nested.is_dir() and any(nested.glob("*.png")):
        return f"Images_{split}: OK ({len(list(nested.glob('*.png')))} images)"
    pngs = list(base.glob("*.png"))
    if pngs:                                   # images one level too high -> push down
        nested.mkdir(parents=True, exist_ok=True)
        for p in pngs:
            p.rename(nested / p.name)
        return f"Images_{split}: fixed nesting ({len(pngs)} images)"
    for d in base.rglob("*"):                  # sitting in some other subdir
        if d.is_dir() and any(d.glob("*.png")):
            if d != nested:
                if nested.exists(): shutil.rmtree(nested)
                d.rename(nested)
            return f"Images_{split}: located + nested"
    return f"Images_{split}: NO PNGs FOUND"


print("\n=== zips present ===")
for z in sorted(DEST.glob("*.zip")):
    print(f"  {z.name}  ({z.stat().st_size/1e9:.2f} GB)")

for split in ("train", "val"):
    zi = DEST / f"Images_{split}.zip"
    if zi.exists():
        print(f"\nExtracting {zi.name} ...")
        extract(zi, DEST / f"Images_{split}")
    za = DEST / f"Annotations_{split}.zip"
    if za.exists():
        print(f"Extracting {za.name} ...")
        extract(za, DEST / f"Annotations_{split}")

print("\n=== fixing image nesting ===")
for split in ("train", "val"):
    print(" ", ensure_nested(split))

print("\n=== verification ===")
for name, p in {"VRSBench_EVAL_vqa.json": DEST/"VRSBench_EVAL_vqa.json",
                "VRSBench_train.json":    DEST/"VRSBench_train.json"}.items():
    print(("OK  " if p.exists() else "MISSING "), name)
for split in ("val", "train"):
    d = DEST / f"Images_{split}" / f"Images_{split}"
    n = len(list(d.glob("*.png"))) if d.is_dir() else 0
    print((f"OK  {n} images" if n else "MISSING "), f"Images_{split}/Images_{split}")
print("\nDone.")

"""
STEP 2 — Exploratory data analysis (EDA).
Question-type histogram, top answers per type, image size sample.
Saves: results/eda_report.txt and results/eda_question_types.png

Run:   python step02_eda.py
Time:  ~30-60 seconds (CPU; samples 100 images for sizes)
"""
import json, collections, random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "VRSBench_data"
OUT  = Path(__file__).resolve().parent / "results"

def main():
    OUT.mkdir(exist_ok=True)
    vqa = json.load(open(DATA / "VRSBench_EVAL_vqa.json", encoding="utf-8"))
    lines = [f"Total VQA eval items: {len(vqa)}", ""]

    types = collections.Counter(x["type"] for x in vqa)
    by_type = collections.defaultdict(collections.Counter)
    for x in vqa:
        by_type[x["type"]][str(x["ground_truth"]).strip().lower()] += 1

    lines.append(f"{'question type':22s}{'count':>8s}   top-3 answers")
    lines.append("-" * 80)
    for t, n in types.most_common():
        top3 = ", ".join(f"'{a}'({c})" for a, c in by_type[t].most_common(3))
        lines.append(f"{t:22s}{n:8d}   {top3}")

    # image size sample
    img_dir = DATA / "Images_val" / "Images_val"
    sample = random.Random(0).sample(sorted(img_dir.glob("*.png")), 100)
    try:
        from PIL import Image
        sizes = collections.Counter(Image.open(p).size for p in sample)
        lines += ["", "Image sizes (100-image sample):"]
        lines += [f"  {w}x{h}: {c}" for (w, h), c in sizes.most_common()]
    except ImportError:
        lines.append("\n(pillow not installed — skipped image size check)")

    report = "\n".join(lines)
    print(report)
    (OUT / "eda_report.txt").write_text(report, encoding="utf-8")

    # histogram plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels, counts = zip(*types.most_common())
        plt.figure(figsize=(10, 4))
        plt.bar(labels, counts)
        plt.xticks(rotation=45, ha="right"); plt.ylabel("count")
        plt.title("VRSBench VQA eval — question types")
        plt.tight_layout()
        plt.savefig(OUT / "eda_question_types.png", dpi=120)
        print(f"\nSaved plot: {OUT / 'eda_question_types.png'}")
    except ImportError:
        print("\n(matplotlib not installed — skipped plot)")

if __name__ == "__main__":
    main()

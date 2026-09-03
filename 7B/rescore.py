"""
rescore.py — Re-grade an EXISTING predictions file with the upgraded step05
harness. Does NOT run the model, so it takes ~1 second and needs no GPU.
Use it to apply the new normalisation (number words, plurals) to results you
already have.

JUPYTERLAB:  set NOTEBOOK_ARGS below, then in a cell:  %run rescore.py
TERMINAL:    python rescore.py --pred preds_finetuned_v2.json

Prints the before/after overall accuracy and rewrites the file with the new
'scores' block (a .bak copy of the original is kept).
"""
import argparse, json, shutil, sys
from pathlib import Path
from step05_eval_harness import score_items, print_scores

# Aspire 7 / OneDrive location (matches api_judge.py). Edit if your folder differs.
BASE = Path(__file__).resolve().parent
OUT = Path(__file__).resolve().parent / "results"

# JUPYTER: which file to re-score (change and re-run for each one)
NOTEBOOK_ARGS = ["--pred", "preds_finetuned_v2.json"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="preds_finetuned_v2.json")
    in_notebook = "ipykernel" in sys.modules
    return ap.parse_args(NOTEBOOK_ARGS if in_notebook else None)


def main():
    args = parse_args()
    path = OUT / args.pred
    data = json.load(open(path, encoding="utf-8"))
    items = data["items"]

    old = data.get("scores", {}).get("overall_acc", None)
    scores = score_items(items)           # re-grades with the new harness
    data["scores"] = scores

    print_scores(scores, label=args.pred)
    if old is not None:
        print(f"\noverall: {100*old:.1f}%  ->  {100*scores['overall_acc']:.1f}%  "
              f"({100*(scores['overall_acc']-old):+.1f} pts)")

    shutil.copy(path, str(path) + ".bak")  # keep the original
    json.dump(data, open(path, "w"), indent=1)
    print(f"Rewrote {path}  (original saved as {path.name}.bak)")


if __name__ == "__main__":
    main()

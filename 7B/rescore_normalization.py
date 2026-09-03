"""
rescore_normalization.py — re-grade an existing predictions file with the
synonym-aware scorer (step05_eval_harness_normalization). Runs NO model, needs
no GPU, ~1 second. Works on ANY prediction file — 3B, 7B or API.

Writes a NEW file  <pred>_normalized.json  (originals are left untouched) and
prints the before -> after overall + per-type accuracy.

JUPYTER:  set NOTEBOOK_ARGS, then  %run rescore_normalization.py
TERMINAL: python rescore_normalization.py --pred preds_zeroshot.json
"""
import argparse, json, sys
from pathlib import Path
from step05_eval_harness_normalization import score_items, print_scores

OUT = Path(__file__).resolve().parent / "results"

# JUPYTER: which file to re-score (change and re-run for each one)
NOTEBOOK_ARGS = ["--pred", "preds_zeroshot.json"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="preds_zeroshot.json")
    in_notebook = "ipykernel" in sys.modules
    return ap.parse_args(NOTEBOOK_ARGS if in_notebook else None)


def main():
    args = parse_args()
    path = OUT / args.pred
    data = json.load(open(path, encoding="utf-8"))
    items = data["items"]

    old = data.get("scores", {}).get("overall_acc", None)
    scores = score_items(items)               # re-grade with the synonym-aware scorer
    print_scores(scores, label=args.pred + "  (synonym-normalized)")
    if old is not None:
        print(f"\noverall: {100*old:.1f}%  ->  {100*scores['overall_acc']:.1f}%  "
              f"({100*(scores['overall_acc']-old):+.1f} pts)")

    out = dict(data)
    out["scores"] = scores
    out_name = args.pred.replace(".json", "_normalized.json")
    json.dump(out, open(OUT / out_name, "w"), indent=1)
    print(f"Saved: {OUT / out_name}   (original {args.pred} untouched)")


if __name__ == "__main__":
    main()

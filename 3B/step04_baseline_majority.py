"""
STEP 4 — Trivial baselines (the floor any model must beat).
Majority-answer-per-question-type + random baseline.
Saves: results/baseline_majority.json

Run:   python step04_baseline_majority.py
Time:  ~15 seconds (CPU)
"""
import json, collections, random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "VRSBench_data"
OUT  = Path(__file__).resolve().parent / "results"

def norm(a): return str(a).strip().lower().rstrip(".")

def main():
    OUT.mkdir(exist_ok=True)
    vqa = json.load(open(DATA / "VRSBench_EVAL_vqa.json", encoding="utf-8"))

    by_type = collections.defaultdict(list)
    for x in vqa:
        by_type[x["type"]].append(norm(x["ground_truth"]))
    majority = {t: collections.Counter(a).most_common(1)[0][0] for t, a in by_type.items()}

    # majority baseline
    per_type, total_correct = {}, 0
    for t, answers in by_type.items():
        c = sum(1 for a in answers if a == majority[t])
        per_type[t] = {"n": len(answers), "acc": c / len(answers), "majority_answer": majority[t]}
        total_correct += c
    maj_acc = total_correct / len(vqa)

    # random baseline (pick a random gt answer of the same type)
    rng = random.Random(0)
    rand_correct = sum(1 for x in vqa if rng.choice(by_type[x["type"]]) == norm(x["ground_truth"]))
    rand_acc = rand_correct / len(vqa)

    print(f"{'question type':22s}{'count':>8s}{'majority-acc':>14s}   most-common answer")
    print("-" * 72)
    for t, d in sorted(per_type.items(), key=lambda kv: -kv[1]["n"]):
        print(f"{t:22s}{d['n']:8d}{100*d['acc']:13.1f}%   '{d['majority_answer']}'")
    print("-" * 72)
    print(f"OVERALL majority baseline: {100*maj_acc:.1f}%   random baseline: {100*rand_acc:.1f}%")

    json.dump({"baseline": "majority-per-type", "overall_acc": maj_acc,
               "random_acc": rand_acc, "per_type": per_type},
              open(OUT / "baseline_majority.json", "w"), indent=2)
    print(f"\nSaved: {OUT / 'baseline_majority.json'}")

if __name__ == "__main__":
    main()

"""
STEP 10 — Novelty B: answer-evidence consistency check (hallucination flag).
Compares each answer against the detector's boxes: claimed class present?
claimed count plausible? Flags inconsistent answers -> abstain.
Reports RQ2 KPIs: flag rate, selective accuracy, flag precision & recall.
Needs results/preds_*.json (step 6/8) and results/detections.json (step 7).
Saves: results/consistency.json

Run:   python step10_consistency_check.py --pred preds_grounded.json
Time:  ~10 seconds (CPU)
"""
import argparse, json, re
from pathlib import Path
from step07_grounding_detector import OBJECT_VOCAB

OUT = Path(__file__).resolve().parent / "results"

WORD2NUM = {"zero": 0, "no": 0, "none": 0, "one": 1, "a": 1, "an": 1, "two": 2,
            "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}

def claimed_count(answer):
    a = answer.lower()
    m = re.search(r"\b(\d+)\b", a)
    if m:
        return int(m.group(1))
    for w, n in WORD2NUM.items():
        if re.search(rf"\b{w}\b", a):
            return n
    return None

def claimed_classes(answer):
    a = answer.lower()
    return [w for w in OBJECT_VOCAB if re.search(rf"\b{re.escape(w)}s?\b", a)]

def check(item, dets):
    """Returns (consistent: bool, reason: str)."""
    det_classes = {d["label"].lower() for d in dets}
    # class check: answer names an object the detector never saw
    for cls in claimed_classes(item["pred"]):
        if not any(cls in dc or dc in cls for dc in det_classes):
            return False, f"answer claims '{cls}' but detector found none"
    # count check (counting questions only)
    if "count" in item["type"].lower() or "how many" in item["question"].lower():
        n_claim = claimed_count(item["pred"])
        if n_claim is not None and abs(n_claim - len(dets)) > max(2, 0.5 * len(dets)):
            return False, f"answer says {n_claim}, detector found {len(dets)} boxes"
    return True, ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="preds_grounded.json")
    args = ap.parse_args()

    det_path = OUT / "detections.json"
    if not det_path.exists():
        raise SystemExit("results/detections.json missing — run step07 first.")
    detections = json.load(open(det_path, encoding="utf-8"))
    data = json.load(open(OUT / args.pred, encoding="utf-8"))

    flagged, results = 0, []
    kept_correct = kept_total = 0
    for x in data["items"]:
        dets = detections.get(f"{x['image_id']}||{x['question']}", [])
        consistent, reason = check(x, dets)
        flagged += not consistent
        if consistent:
            kept_total += 1
            kept_correct += bool(x["correct"])
        results.append({**x, "consistent": consistent, "flag_reason": reason})

    n = len(results)
    base_acc = data["scores"]["overall_acc"]
    sel_acc  = kept_correct / kept_total if kept_total else 0.0
    print(f"Items checked:            {n}")
    print(f"Flagged as inconsistent:  {flagged} ({100*flagged/n:.1f}%) -> abstain")
    print(f"Accuracy before check:    {100*base_acc:.1f}%")
    print(f"Accuracy on kept answers: {100*sel_acc:.1f}%  (coverage {100*kept_total/n:.1f}%)")

    # RQ2 KPIs — treat the flag as a "wrong answer" detector:
    #   precision = of flagged answers, how many were actually wrong
    #   recall    = of all wrong answers, how many did the flag catch
    wrong_flagged = sum(1 for r in results if not r["consistent"] and not r["correct"])
    total_wrong   = sum(1 for r in results if not r["correct"])
    precision = wrong_flagged / flagged if flagged else 0.0
    recall    = wrong_flagged / total_wrong if total_wrong else 0.0
    print("\nRQ2 flag quality (vs ground truth):")
    if flagged:
        print(f"  Flag precision: {wrong_flagged}/{flagged} flagged were truly wrong "
              f"({100*precision:.1f}%)")
    else:
        print("  Flag precision: n/a (nothing flagged)")
    if total_wrong:
        print(f"  Flag recall:    {wrong_flagged}/{total_wrong} of all wrong answers "
              f"caught ({100*recall:.1f}%)")
    else:
        print("  Flag recall: n/a (no wrong answers)")

    json.dump({"pred_file": args.pred, "n": n, "flagged": flagged,
               "acc_before": base_acc, "selective_acc": sel_acc,
               "coverage": kept_total / n if n else 0,
               "flag_precision": precision, "flag_recall": recall,
               "total_wrong": total_wrong, "wrong_flagged": wrong_flagged,
               "items": results},
              open(OUT / "consistency.json", "w"), indent=1)
    print(f"Saved: {OUT / 'consistency.json'}")

if __name__ == "__main__":
    main()

"""
STEP 11 — Aggregate all results into one summary table.
Reads every results/*.json produced by earlier steps.
Saves: results/RESULTS_SUMMARY.md

Run:   python step11_make_report.py
Time:  ~5 seconds (CPU)
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results"

def get(path):
    p = OUT / path
    return json.load(open(p, encoding="utf-8")) if p.exists() else None

def main():
    lines = ["# FM-12a Results Summary", ""]
    rows = [("Condition", "n", "Overall acc", "Notes")]

    if (b := get("baseline_majority.json")):
        rows.append(("Majority-per-type baseline", "37409", f"{100*b['overall_acc']:.1f}%",
                     f"random: {100*b['random_acc']:.1f}%"))
    if (z := get("preds_zeroshot.json")):
        rows.append((f"Zero-shot ({z['meta']['model']})", str(z['meta']['n']),
                     f"{100*z['scores']['overall_acc']:.1f}%", "no grounding"))
    if (g := get("preds_grounded.json")):
        rows.append((f"Grounded pipeline ({g['meta']['model']})", str(g['meta']['n']),
                     f"{100*g['scores']['overall_acc']:.1f}%", "detector boxes in prompt"))
    if (gr := get("grounding_scores.json")):
        rows.append(("Grounding DINO (referring)", str(gr["n"]),
                     f"Acc@0.5 {100*gr['acc@0.5']:.1f}%", f"Acc@0.7 {100*gr['acc@0.7']:.1f}%"))
    if (c := get("calibration.json")):
        rows.append(("Calibrated + abstention", str(c["n"]),
                     f"{100*c['selective_acc']:.1f}% selective",
                     f"cov {100*c['coverage']:.0f}%, ECE {c['ece_before']:.3f}->{c['ece_after']:.3f}, T={c['temperature']:.2f}"))
    if (k := get("consistency.json")):
        extra = ""
        if "flag_precision" in k:
            extra = f", flag P {100*k['flag_precision']:.0f}% / R {100*k['flag_recall']:.0f}%"
        rows.append(("Consistency-checked", str(k["n"]),
                     f"{100*k['selective_acc']:.1f}% on kept",
                     f"flagged {k['flagged']} ({100*k['flagged']/k['n']:.1f}%){extra}"))
    if (s := get("shift_rsvqa.json")):
        rows.append(("Shift test (RSVQA-LR)", str(s["meta"]["n"]),
                     f"{100*s['scores']['overall_acc']:.1f}%",
                     f"ECE raw {s['ece_raw']:.3f} -> frozen-T {s['ece_frozen']:.3f} "
                     f"(oracle {s['ece_oracle']:.3f})"))

    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    lines.append("| " + " | ".join(rows[0][i].ljust(widths[i]) for i in range(4)) + " |")
    lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows[1:]:
        lines.append("| " + " | ".join(r[i].ljust(widths[i]) for i in range(4)) + " |")

    # per-type breakdown of the best available run
    best = get("preds_grounded.json") or get("preds_zeroshot.json")
    if best:
        lines += ["", f"## Per-type accuracy ({best['meta']['condition']})", ""]
        for t, d in sorted(best["scores"]["per_type"].items()):
            lines.append(f"- {t}: {100*d['acc']:.1f}% (n={d['n']})")

    report = "\n".join(lines)
    print(report)
    (OUT / "RESULTS_SUMMARY.md").write_text(report, encoding="utf-8")
    print(f"\nSaved: {OUT / 'RESULTS_SUMMARY.md'}")

if __name__ == "__main__":
    main()

"""
STEP 9 — Novelty A: confidence calibration + selective answering.
Reads a predictions file (step 6 or 8), computes ECE, reliability diagram,
temperature scaling, abstention threshold and risk-coverage curve.
Saves: results/calibration.json, reliability_diagram.png, risk_coverage.png

Run:   python step09_calibration.py --pred preds_zeroshot.json
Time:  ~30 seconds (CPU)
"""
import argparse, json, math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results"

def ece(confs, corrects, n_bins=10):
    bins = [[] for _ in range(n_bins)]
    for c, ok in zip(confs, corrects):
        bins[min(int(c * n_bins), n_bins - 1)].append((c, ok))
    total, err = len(confs), 0.0
    stats = []
    for b in bins:
        if not b:
            stats.append(None); continue
        avg_c = sum(c for c, _ in b) / len(b)
        acc   = sum(ok for _, ok in b) / len(b)
        err  += len(b) / total * abs(avg_c - acc)
        stats.append({"n": len(b), "avg_conf": avg_c, "acc": acc})
    return err, stats

def apply_temperature(conf, T):
    """Scale confidence in logit space: sigmoid(logit(c)/T)."""
    c = min(max(conf, 1e-6), 1 - 1e-6)
    z = math.log(c / (1 - c)) / T
    return 1 / (1 + math.exp(-z))

def fit_temperature(confs, corrects):
    """Grid-search T minimising NLL (no scipy needed)."""
    def nll(T):
        s = 0.0
        for c, ok in zip(confs, corrects):
            p = min(max(apply_temperature(c, T), 1e-6), 1 - 1e-6)
            s += -math.log(p) if ok else -math.log(1 - p)
        return s / len(confs)
    Ts = [x / 20 for x in range(2, 101)]          # 0.1 .. 5.0
    return min(Ts, key=nll)

def risk_coverage(confs, corrects):
    order = sorted(range(len(confs)), key=lambda i: -confs[i])
    pts, correct_so_far = [], 0
    for k, i in enumerate(order, 1):
        correct_so_far += corrects[i]
        pts.append({"coverage": k / len(confs),
                    "selective_acc": correct_so_far / k,
                    "threshold": confs[i]})
    return pts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="preds_zeroshot.json")
    ap.add_argument("--target-acc", type=float, default=0.85,
                    help="pick abstention threshold aiming at this selective accuracy")
    args = ap.parse_args()

    data = json.load(open(OUT / args.pred, encoding="utf-8"))
    items = data["items"]
    confs    = [x["conf"] for x in items]
    corrects = [bool(x["correct"]) for x in items]

    e_before, stats_before = ece(confs, corrects)
    T = fit_temperature(confs, corrects)
    confs_T = [apply_temperature(c, T) for c in confs]
    e_after, stats_after = ece(confs_T, corrects)
    print(f"ECE before temperature scaling: {e_before:.4f}")
    print(f"Fitted temperature T = {T:.2f}")
    print(f"ECE after  temperature scaling: {e_after:.4f}")

    rc = risk_coverage(confs_T, corrects)
    # choose threshold: highest coverage whose selective accuracy >= target
    chosen = next((p for p in rc[::-1] if p["selective_acc"] >= args.target_acc), rc[0])
    print(f"\nAbstention threshold {chosen['threshold']:.3f} -> "
          f"coverage {100*chosen['coverage']:.1f}%, "
          f"selective accuracy {100*chosen['selective_acc']:.1f}% "
          f"(target {100*args.target_acc:.0f}%)")

    json.dump({"pred_file": args.pred, "n": len(items),
               "ece_before": e_before, "temperature": T, "ece_after": e_after,
               "abstention_threshold": chosen["threshold"],
               "coverage": chosen["coverage"],
               "selective_acc": chosen["selective_acc"],
               "risk_coverage": rc[::max(1, len(rc)//100)]},
              open(OUT / "calibration.json", "w"), indent=1)
    print(f"Saved: {OUT / 'calibration.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # reliability diagram
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for a, stats, title in ((ax[0], stats_before, f"Before (ECE {e_before:.3f})"),
                                (ax[1], stats_after,  f"After T={T:.2f} (ECE {e_after:.3f})")):
            xs = [(i + 0.5) / 10 for i, s in enumerate(stats) if s]
            ys = [s["acc"] for s in stats if s]
            a.bar(xs, ys, width=0.09)
            a.plot([0, 1], [0, 1], "k--")
            a.set_xlabel("confidence"); a.set_ylabel("accuracy"); a.set_title(title)
        plt.tight_layout(); plt.savefig(OUT / "reliability_diagram.png", dpi=120)
        # risk-coverage curve
        plt.figure(figsize=(5, 4))
        plt.plot([p["coverage"] for p in rc], [p["selective_acc"] for p in rc])
        plt.axhline(args.target_acc, ls="--", c="r")
        plt.xlabel("coverage"); plt.ylabel("selective accuracy")
        plt.title("Risk-coverage"); plt.tight_layout()
        plt.savefig(OUT / "risk_coverage.png", dpi=120)
        print(f"Saved: reliability_diagram.png, risk_coverage.png")
    except ImportError:
        print("(matplotlib not installed — skipped plots)")

if __name__ == "__main__":
    main()

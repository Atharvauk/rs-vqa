"""
STEP 5 — Evaluation harness (imported by steps 6, 8, 9, 10).
Answer normalisation, soft exact-match, per-type scoring, Accuracy@IoU.
Run as a script it self-tests.

v2 normalisation upgrade (no new dependencies):
  * number words <-> digits   ('two' == '2', 'ten' == '10')
  * plural -> singular         ('airplanes' == 'airplane', 'trees' == 'tree')
  * drops leading articles     ('a windmill' == 'windmill')
These recover CORRECT answers that differ only in format; they never turn a
wrong answer into a right one (both sides get the identical normalisation).

Run:   python step05_eval_harness.py
Time:  ~5 seconds (CPU)
"""
import collections, re

# ---- word <-> digit table (covers the small counts VRSBench uses) ----------
_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100",
}
_ARTICLES = {"a", "an", "the"}


def _singular(w):
    """Very light plural -> singular. Symmetric, so it is safe to over-apply."""
    if len(w) <= 3:                       # protect 'bus', 'yes', 'no'
        return w
    if w.endswith(("ss", "us", "is")):    # 'grass', 'status', 'axis'
        return w
    if w.endswith("ies") and len(w) > 4:  # 'facilities' -> 'facility'
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "xes", "zes", "ses")):  # 'boxes' -> 'box'
        return w[:-2]
    if w.endswith("es") and len(w) > 4:   # 'airplanes' -> 'airplane'
        return w[:-1]
    if w.endswith("s"):                   # 'cars' -> 'car'
        return w[:-1]
    return w


def norm(a):
    """Canonical form: lowercase, de-punctuate, digits, singular, no articles."""
    toks = str(a).strip().lower().rstrip(".").split()
    out = []
    for w in toks:
        w = w.strip(".,!?;:\"'()[]")
        if not w or w in _ARTICLES:
            continue
        w = _NUM.get(w, w)                # number word -> digit
        w = _singular(w)                  # plural -> singular
        out.append(w)
    return " ".join(out)


def is_match(pred, gt):
    """Soft exact-match: equal, or one appears as a whole word/phrase in the other."""
    p, g = norm(pred), norm(gt)
    if p == g:
        return True
    if g and re.search(rf"(?<!\w){re.escape(g)}(?!\w)", p):
        return True
    if p and re.search(rf"(?<!\w){re.escape(p)}(?!\w)", g):
        return True
    return False


def score_items(items):
    """items: [{'type':..,'gt':..,'pred':..}] -> dict with overall + per-type accuracy."""
    per = collections.defaultdict(lambda: [0, 0])
    correct = 0
    for x in items:
        ok = is_match(x["pred"], x["gt"])
        x["correct"] = bool(ok)
        correct += ok
        per[x["type"]][1] += 1
        per[x["type"]][0] += ok
    return {"n": len(items),
            "overall_acc": correct / len(items) if items else 0.0,
            "per_type": {t: {"n": n, "acc": c / n} for t, (c, n) in per.items()}}


def acc_at_iou(pairs, thresholds=(0.5, 0.7)):
    """pairs: [(pred_box_xyxy, gt_box_xyxy)] in the SAME coordinate scale."""
    from step03_coord_utils import iou
    out = {}
    for t in thresholds:
        hits = sum(1 for p, g in pairs if p is not None and iou(p, g) >= t)
        out[t] = hits / len(pairs) if pairs else 0.0
    return out


def print_scores(scores, label=""):
    print(f"\n{'question type':22s}{'n':>7s}{'accuracy':>10s}   {label}")
    print("-" * 50)
    for t, d in sorted(scores["per_type"].items()):
        print(f"{t:22s}{d['n']:7d}{100*d['acc']:9.1f}%")
    print("-" * 50)
    print(f"{'OVERALL':22s}{scores['n']:7d}{100*scores['overall_acc']:9.1f}%")


if __name__ == "__main__":
    # original tests
    assert is_match("Yellow.", "yellow")
    assert is_match("there are 3 cars", "3")
    assert not is_match("no", "yes")
    # new tests: number words and plurals
    assert is_match("Two", "2")
    assert is_match("2", "two")
    assert is_match("Airplanes", "airplane")
    assert is_match("trees", "Tree")
    assert is_match("a windmill", "windmill")
    assert not is_match("three", "2")        # wrong count stays wrong
    assert not is_match("yes", "no")
    s = score_items([{"type": "color", "gt": "red", "pred": "Red."},
                     {"type": "count", "gt": "2", "pred": "two"},
                     {"type": "color", "gt": "blue", "pred": "green"}])
    assert abs(s["overall_acc"] - 2/3) < 1e-9
    print("Eval-harness v2 self-tests PASSED.")

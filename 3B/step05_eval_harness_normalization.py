"""
step05_eval_harness_normalization.py — scorer WITH a remote-sensing synonym map
on top of the plural / number-word normalisation. Credits answers that mean the
same thing but use different vocabulary, e.g. 'wind turbine' == 'windmill',
'forest' == 'trees', 'vehicle' == 'car', 'aircraft' == 'airplane'.

IMPORTANT: this runs NO model. It only RE-GRADES existing prediction files, so it
applies equally to the 3B, 7B and API results. Same public functions as step05
(score_items, print_scores, is_match, norm) — used by rescore_normalization.py.
"""
import collections, re

# ---- number words <-> digits ------------------------------------------------
_NUM = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6",
        "seven":"7","eight":"8","nine":"9","ten":"10","eleven":"11","twelve":"12",
        "thirteen":"13","fourteen":"14","fifteen":"15","sixteen":"16","seventeen":"17",
        "eighteen":"18","nineteen":"19","twenty":"20","thirty":"30","forty":"40",
        "fifty":"50","sixty":"60","seventy":"70","eighty":"80","ninety":"90","hundred":"100"}
_ARTICLES = {"a", "an", "the"}

# ---- remote-sensing synonym groups: every variant maps to one canonical form.
#      Keys are written in the already-normalised (lowercase, singular) form.
_SYN_GROUPS = {
    "airplane":     ["airplane", "aeroplane", "plane", "aircraft", "jet", "airliner"],
    "windmill":     ["windmill", "wind turbine", "wind mill", "turbine"],
    "tree":         ["tree", "forest", "woodland", "wood", "grove"],
    "car":          ["car", "vehicle", "automobile", "auto"],
    "ship":         ["ship", "boat", "vessel"],
    "pool":         ["pool", "swimming pool"],
    "storage tank": ["storage tank", "tank", "oil tank", "fuel tank"],
    "roundabout":   ["roundabout", "traffic circle", "rotary"],
    "playground":   ["playground", "play field", "play area"],
    "large":        ["large", "big", "larger", "bigger"],
    "small":        ["small", "little", "smaller", "tiny"],
    "yes":          ["yes", "yeah", "yep", "true", "correct"],
    "no":           ["no", "nope", "false", "incorrect"],
}
_SYN = {v: canon for canon, variants in _SYN_GROUPS.items() for v in variants}


def _singular(w):
    """Light plural -> singular. Symmetric, so safe to over-apply."""
    if len(w) <= 3:
        return w
    if w.endswith(("ss", "us", "is")):
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "xes", "zes", "ses")):
        return w[:-2]
    if w.endswith("es") and len(w) > 4:
        return w[:-1]
    if w.endswith("s"):
        return w[:-1]
    return w


def norm(a):
    """lowercase -> de-punctuate -> digits -> singular -> drop articles -> synonym canonical."""
    toks = str(a).strip().lower().rstrip(".").split()
    out = []
    for w in toks:
        w = w.strip(".,!?;:\"'()[]")
        if not w or w in _ARTICLES:
            continue
        w = _NUM.get(w, w)
        w = _singular(w)
        out.append(w)
    phrase = " ".join(out)
    return _SYN.get(phrase, phrase)          # map whole phrase to its canonical, if known


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
    assert is_match("Airplanes", "airplane")
    assert is_match("wind turbine", "windmill")
    assert is_match("Forest", "trees")
    assert is_match("vehicle", "cars")
    assert is_match("Two", "2")
    assert not is_match("windmill", "airplane")
    assert not is_match("yes", "no")
    print("normalization eval-harness self-tests PASSED.")

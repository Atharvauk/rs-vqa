"""

STEP 12 — RQ3 distribution-shift test: does the VRSBench-fitted calibration

survive on an UNSEEN dataset (RSVQA-LR)?
 
Protocol:

  1. Temperature T was fitted on VRSBench predictions (step 9) — FROZEN here.

  2. Run the same zero-shot VLM on RSVQA-LR test questions, restricted to the

     question types both datasets share (presence, count) so it is fair.

  3. Report ECE raw, ECE with the FROZEN T (the claim), and ECE with a T

     refitted on RSVQA itself (oracle — diagnostic only).
 
RSVQA-LR is downloaded AUTOMATICALLY from Zenodo (record 6344333) into

  pipeline_7b/RSVQA_LR_data/

Needs results/calibration.json from step 9 (that's where the frozen T lives).

Saves: results/shift_rsvqa.json
 
REAL RUN:

    python step12_rsvqa_shift.py --limit 1000 --load-4bit

DRY TEST (no GPU, no download):

    python step12_rsvqa_shift.py --dry-run --limit 200

"""

import argparse, json, random, sys, time, zipfile, urllib.request

from pathlib import Path
 
from step05_eval_harness import score_items, print_scores

from step09_calibration import ece, apply_temperature, fit_temperature

import step06_zeroshot_vqa as zs
 
BASE = Path(__file__).resolve().parent

DATA = BASE.parent / "RSVQA_LR_data"

OUT  = Path(__file__).resolve().parent / "results"

OUT.mkdir(parents=True, exist_ok=True)
 
ZENODO_RECORD = "6344333"                 # RSVQA-LR (Lobry et al.)

SHARED_TYPES  = ("presence", "count")     # types VRSBench and RSVQA share
 
NOTEBOOK_ARGS = ["--limit", "200", "--load-4bit"]

# dry test:   NOTEBOOK_ARGS = ["--dry-run", "--limit", "200"]
 
 
# ---------------------------------------------------------------- download

def _download(url, dest):

    """Stream a file to disk with a progress line. Skips if already complete."""

    if dest.exists() and dest.stat().st_size > 0:

        print(f"      have {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")

        return dest

    print(f"      downloading {dest.name} ...")

    tmp, t0 = dest.with_suffix(dest.suffix + ".part"), time.perf_counter()

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:

        total, got = int(r.headers.get("Content-Length") or 0), 0

        while True:

            block = r.read(1 << 20)

            if not block:

                break

            f.write(block)

            got += len(block)

            if total:

                print(f"\r        {got/1e6:8.1f}/{total/1e6:.1f} MB "

                      f"({100*got/total:5.1f}%)", end="", flush=True)

            else:

                print(f"\r        {got/1e6:8.1f} MB", end="", flush=True)

    print(f"   done in {time.perf_counter()-t0:.0f}s")

    tmp.replace(dest)

    return dest
 
 
def find_images_dir():

    """Locate the folder holding the .tif images (handles nested zip folders)."""

    if not DATA.exists():

        return None

    d = DATA / "Images_LR"

    if d.is_dir() and any(d.glob("*.tif")):

        return d

    for p in DATA.rglob("*"):

        if p.is_dir() and any(p.glob("*.tif")):

            return p

    return None
 
 
def ensure_rsvqa():

    """Download + unpack RSVQA-LR into DATA if it isn't already there."""

    qf, af = DATA / "LR_split_test_questions.json", DATA / "LR_split_test_answers.json"

    if qf.exists() and af.exists() and find_images_dir():

        print("      RSVQA-LR already present")

        return

    DATA.mkdir(parents=True, exist_ok=True)
 
    # map {filename: real download URL} using the link the API gives per file

    url_by_name = {}

    try:

        req = urllib.request.Request(

            f"https://zenodo.org/api/records/{ZENODO_RECORD}",

            headers={"User-Agent": "Mozilla/5.0"})

        with urllib.request.urlopen(req, timeout=60) as r:

            files = json.load(r).get("files", [])

        for fobj in files:

            name = fobj.get("key") or fobj.get("filename")

            links = fobj.get("links", {}) or {}

            link = links.get("self") or links.get("download") or links.get("content")

            if name:

                url_by_name[name] = link or (

                    f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}/content")

        print(f"      Zenodo record lists {len(url_by_name)} files")

    except Exception as e:

        print(f"      (could not list Zenodo files: {e}; using default names)")

        for name in ["LR_split_test_questions.json",

                     "LR_split_test_answers.json", "Images_LR.zip"]:

            url_by_name[name] = (

                f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}/content")
 
    wanted = [n for n in url_by_name

              if (n.lower().endswith(".json") and "test" in n.lower()

                  and ("question" in n.lower() or "answer" in n.lower()))

              or (n.lower().endswith(".zip") and "image" in n.lower())]

    if not wanted:

        raise SystemExit(f"No matching files on Zenodo record {ZENODO_RECORD}. "

                         "Download manually from https://zenodo.org/records/6344333")
 
    for n in wanted:

        dest = _download(url_by_name[n], DATA / n)

        if n.lower().endswith(".zip"):

            print(f"      extracting {n} ...")

            with zipfile.ZipFile(dest) as z:

                z.extractall(DATA)

            print(f"      extracted to {DATA}")
 
 
# ---------------------------------------------------------------- data

def load_rsvqa(limit, types, seed=0):

    qf, af = DATA / "LR_split_test_questions.json", DATA / "LR_split_test_answers.json"

    if not (qf.exists() and af.exists()):

        raise SystemExit(f"RSVQA-LR question/answer files missing under {DATA}.")

    questions = json.load(open(qf, encoding="utf-8"))["questions"]

    answers   = json.load(open(af, encoding="utf-8"))["answers"]

    ans_by_q  = {a["question_id"]: str(a["answer"])

                 for a in answers if a.get("active", True)}

    items = [{"image_id": f"{q['img_id']}.tif", "question": q["question"],

              "type": q["type"], "ground_truth": ans_by_q.get(q["id"], "")}

             for q in questions

             if q.get("active", True) and q["type"] in types and q["id"] in ans_by_q]

    random.Random(seed).shuffle(items)

    return items[:limit] if limit else items
 
 
def fake_rsvqa(limit, seed=0):

    """Synthetic presence/count items for --dry-run."""

    rng, items = random.Random(seed), []

    for i in range(limit):

        if rng.random() < 0.6:

            items.append({"image_id": f"fake_{i}.tif", "type": "presence",

                          "question": "Is a road present in the image?",

                          "ground_truth": rng.choice(["yes", "no"])})

        else:

            items.append({"image_id": f"fake_{i}.tif", "type": "count",

                          "question": "How many buildings are there?",

                          "ground_truth": str(rng.randint(0, 10))})

    return items
 
 
# ---------------------------------------------------------------- args

def parse_args():

    ap = argparse.ArgumentParser()

    ap.add_argument("--limit", type=int, default=200)

    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")

    ap.add_argument("--load-4bit", action="store_true")

    ap.add_argument("--dry-run", action="store_true")

    ap.add_argument("--types", nargs="+", default=list(SHARED_TYPES))

    in_notebook = "ipykernel" in sys.modules

    return ap.parse_args(NOTEBOOK_ARGS if in_notebook else None)
 
 
# ---------------------------------------------------------------- main

def main():

    args = parse_args()

    t_start = time.perf_counter()
 
    # 1) frozen temperature from step 9

    cal_path = OUT / "calibration.json"

    if not cal_path.exists():

        raise SystemExit("results/calibration.json missing — run step09 on the "

                         "VRSBench predictions first (that fits the frozen T).")

    T_frozen = json.load(open(cal_path, encoding="utf-8"))["temperature"]

    print(f"[1/5] Frozen temperature from VRSBench (step 9): T = {T_frozen:.2f}")
 
    # 2) get the dataset

    if args.dry_run:

        items, images_dir = fake_rsvqa(args.limit), DATA / "Images_LR"

        print(f"[2/5] dry-run: {len(items)} synthetic items (no download)")

    else:

        print("[2/5] Ensuring RSVQA-LR is downloaded...")

        ensure_rsvqa()

        images_dir = find_images_dir()

        if images_dir is None:

            raise SystemExit(f"No .tif images found under {DATA} after download.")

        print(f"      images: {images_dir}")

        items = load_rsvqa(args.limit, set(args.types))

        print(f"      loaded {len(items)} items (types: {', '.join(args.types)})")
 
    # 3) run the same VLM on the unseen dataset

    predict = zs.make_predictor(args)

    n = len(items)

    width = len(str(n))

    records, running_correct = [], 0
 
    print(f"\n[3/5] Answering {n} questions on the UNSEEN dataset")

    print(f"{'idx':>{2*width+3}} | {'time':>6} | {'avg':>8} | {'eta':>6} | "

          f"{'acc':>6} | {'conf':>5} | type / gt -> pred")

    print("-" * 100, flush=True)
 
    t_loop = time.perf_counter()

    for i, x in enumerate(items, 1):

        t0 = time.perf_counter()

        ans, conf = predict(images_dir / x["image_id"], x["question"], x["type"])

        dt = time.perf_counter() - t0
 
        ok = zs.quick_correct(ans, x["ground_truth"])

        running_correct += ok

        records.append({"image_id": x["image_id"], "question": x["question"],

                        "type": x["type"], "gt": x["ground_truth"],

                        "pred": ans, "conf": round(conf, 4), "sec": round(dt, 2)})
 
        elapsed = time.perf_counter() - t_loop

        avg = elapsed / i

        eta = avg * (n - i)

        print(f"[{i:>{width}}/{n}] | {dt:5.1f}s | {avg:6.1f}s/q | {zs.fmt_eta(eta)} | "

              f"{100*running_correct/i:5.1f}% | {conf:5.2f} | "

              f"{x['type'][:16]:16} gt='{str(x['ground_truth'])[:14]}' "

              f"-> '{str(ans)[:14]}' {'OK ' if ok else 'X'}", flush=True)
 
    loop_total = time.perf_counter() - t_loop

    print("-" * 100)

    print(f"Finished {n} items in {loop_total/60:.1f} min "

          f"({loop_total/max(n,1):.1f} s/question average)")
 
    # 4) score + calibration under shift

    scores = score_items(records)

    print_scores(scores, label="RSVQA-LR shift test")
 
    confs    = [r["conf"] for r in records]

    corrects = [bool(r["correct"]) for r in records]

    e_raw, _    = ece(confs, corrects)

    e_frozen, _ = ece([apply_temperature(c, T_frozen) for c in confs], corrects)

    T_oracle    = fit_temperature(confs, corrects)

    e_oracle, _ = ece([apply_temperature(c, T_oracle) for c in confs], corrects)
 
    print(f"\n[4/5] RQ3 shift result (lower ECE = better calibrated):")

    print(f"  ECE raw (no scaling):        {e_raw:.4f}")

    print(f"  ECE with FROZEN T={T_frozen:.2f}:     {e_frozen:.4f}   <- the claim")

    print(f"  ECE with oracle T={T_oracle:.2f}:     {e_oracle:.4f}   (refit on RSVQA, diagnostic)")

    verdict = "TRANSFERS" if e_frozen < e_raw else "does NOT transfer"

    print(f"  Verdict: VRSBench calibration {verdict} to RSVQA-LR.")
 
    # 5) save

    json.dump({"meta": {"model": "dry-run" if args.dry_run else args.model,

                        "n": len(records), "condition": "shift_rsvqa_lr",

                        "types": args.types,

                        "total_sec": round(loop_total, 1),

                        "avg_sec_per_q": round(loop_total/max(len(records), 1), 2)},

               "temperature_frozen": T_frozen, "temperature_oracle": T_oracle,

               "ece_raw": e_raw, "ece_frozen": e_frozen, "ece_oracle": e_oracle,

               "scores": scores, "items": records},

              open(OUT / "shift_rsvqa.json", "w"), indent=1)

    print(f"\n[5/5] Saved: {OUT / 'shift_rsvqa.json'}")

    print(f"Total: {(time.perf_counter()-t_start)/60:.1f} min")
 
 
if __name__ == "__main__":

    main()
 
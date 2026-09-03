"""
RUN 0 - run_all.py  —  FM-12a one-shot launcher.

Runs the project end to end, pausing for your choice at each stage:

  1. Library setup            (RUN 1 - setup_libs.py)
  2. Dataset check            -> if missing: download / continue without / abort
  3. Pick what to launch      -> final_dashboard.py  OR  compare_models.py
  4. Launches it in 4-bit
  5. Press Ctrl+C at any time to quit.

Usage (from the FM12A folder):
    python "RUN 0 - run_all.py"
"""
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
PY = sys.executable

SETUP    = BASE / "RUN 1 - setup_libs.py"
DOWNLOAD = BASE / "RUN 2 - optional - download_vrsbench.py"
DATA     = BASE / "VRSBench_data"
DASHBOARD = BASE / "final_dashboard.py"
COMPARE   = BASE / "compare_models.py"


# ---------------------------------------------------------------- helpers
def line(char="─", n=64):
    print(char * n)

def banner(title):
    print()
    line("═")
    print(f"  {title}")
    line("═")

def ask(prompt, choices):
    """choices = {'1': 'label', ...}. Returns the chosen key. Ctrl+C quits."""
    for k, label in choices.items():
        print(f"   [{k}] {label}")
    valid = set(choices)
    while True:
        pick = input(f"{prompt} ").strip().lower()
        if pick in valid:
            return pick
        print(f"   Please enter one of: {', '.join(choices)}")

def run_script(path, extra=None, label=None):
    """Run a python script with the SAME interpreter. Returns exit code."""
    if not path.exists():
        print(f"   ⚠  Not found: {path.name}")
        return 1
    cmd = [PY, str(path)] + (extra or [])
    print(f"\n▶  Running: {label or path.name}")
    line()
    rc = subprocess.run(cmd, cwd=str(BASE)).returncode
    line()
    print(f"✓  Finished ({path.name}) with exit code {rc}")
    return rc


# ---------------------------------------------------------------- dataset check
def dataset_present():
    """Lightweight check for a usable VRSBench dataset in VRSBench_data/."""
    if not DATA.exists():
        return False, "VRSBench_data/ folder not found."
    eval_json = DATA / "VRSBench_EVAL_vqa.json"
    train_json = DATA / "VRSBench_train.json"
    val_imgs = DATA / "Images_val" / "Images_val"
    has_json = eval_json.exists() or train_json.exists()
    has_imgs = val_imgs.exists() and any(val_imgs.glob("*.png"))
    if has_json and has_imgs:
        return True, "Found VRSBench JSON + validation images."
    if has_json:
        return True, "Found VRSBench JSON (images not verified)."
    return False, "No VRSBench JSON / images found in VRSBench_data/."


# ---------------------------------------------------------------- stages
def stage_setup():
    banner("STAGE 1 · Library setup")
    print("This reinstalls PyTorch (CUDA cu128) and all required libraries.")
    pick = ask("Run library setup now?", {
        "1": "Yes, run setup",
        "2": "Skip (libraries already installed)",
    })
    if pick == "1":
        run_script(SETUP, label="RUN 1 - setup_libs.py")
    else:
        print("   Skipped library setup.")

def stage_dataset():
    banner("STAGE 2 · Dataset check")
    ok, msg = dataset_present()
    print(f"   {msg}")
    if ok:
        print("   Dataset is in place — continuing.")
        return
    print("\n   The dataset is NOT present.")
    print("   (Note: the dashboard and compare tools do NOT need the dataset —")
    print("    it is only required for evaluation scripts.)")
    pick = ask("What would you like to do?", {
        "1": "Download the dataset now  (RUN 2 - download_vrsbench.py, ~13 GB)",
        "2": "Continue WITHOUT downloading  (fine for the demo / compare)",
        "3": "Abort",
    })
    if pick == "1":
        run_script(DOWNLOAD, label="RUN 2 - download_vrsbench.py")
    elif pick == "2":
        print("   Continuing without the dataset.")
    else:
        print("   Aborting.")
        sys.exit(0)

def stage_launch():
    banner("STAGE 3 · Launch")
    print("   Choose what to run:\n")
    print("   [1] final_dashboard.py  —  Showcase DEMO of the fine-tuned 7B model.")
    print("        Upload a satellite image, ask a question, get an answer plus the")
    print("        detected objects. Guided coach tour. Best for the live demo / viva.\n")
    print("   [2] compare_models.py   —  COMPARISON tool. Pick 3B or 7B and")
    print("        Before / After fine-tuning to see the answers side by side.")
    print("        Best for demonstrating the effect of fine-tuning.\n")
    pick = ask("Which do you want to launch?", {
        "1": "final_dashboard.py  (demo)",
        "2": "compare_models.py   (comparison)",
        "3": "Abort",
    })
    if pick == "3":
        print("   Aborting.")
        sys.exit(0)

    share = ask("Create a public share link too?", {
        "1": "No, local only  (http://127.0.0.1:7860)",
        "2": "Yes, add a public --share link",
    })
    extra = ["--load-4bit"] + (["--share"] if share == "2" else [])

    target = DASHBOARD if pick == "1" else COMPARE
    print("\n   Launching in 4-bit. The model takes ~30–60 s to load;")
    print("   watch for 'VLM ready' then open http://127.0.0.1:7860")
    print("   (Press Ctrl+C in this window to stop the server.)")
    run_script(target, extra=extra, label=target.name)


# ---------------------------------------------------------------- main
def main():
    banner("FM-12a · Remote-Sensing Visual QA — launcher")
    print("  Student: Atharva Uday Kalase (40509608) · ECS8060 · QUB")
    print("  Press Ctrl+C at any prompt to quit.")
    stage_setup()
    stage_dataset()
    stage_launch()
    print("\n✓  Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹  Interrupted — quitting. Bye!")
        sys.exit(0)

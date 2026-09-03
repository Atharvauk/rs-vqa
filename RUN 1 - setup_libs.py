"""
setup_libs.py — one-shot environment setup for FM12A.

Uninstalls any existing torch, installs the correct cu128 build (for the Blackwell
RTX 5060), installs all required libraries, then verifies CUDA.

Run:
    python setup_libs.py
"""
import subprocess, sys

PY = sys.executable  # use THIS python's pip


def run(cmd, label):
    print("\n" + "=" * 70)
    print(f">>> {label}")
    print("    " + " ".join(cmd))
    print("=" * 70, flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"!! step failed (exit {rc}) — see the error above.")
    return rc


def main():
    # 1) remove any existing torch stack
    run([PY, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"],
        "1/4  Uninstall existing torch / torchvision / torchaudio")

    # 2) install the cu128 build (Blackwell RTX 5060)
    run([PY, "-m", "pip", "install", "torch", "torchvision", "torchaudio",
         "--index-url", "https://download.pytorch.org/whl/cu128"],
        "2/4  Install torch (cu128 — Blackwell GPU)")

    # 3) all other required libraries (from normal PyPI)
    libs = ["transformers", "accelerate", "peft", "bitsandbytes",
            "qwen-vl-utils", "pillow", "gradio", "openai",
            "huggingface_hub", "matplotlib"]
    run([PY, "-m", "pip", "install", "-U", *libs],
        "3/4  Install project libraries")

    # 4) verify CUDA + versions
    print("\n" + "=" * 70)
    print(">>> 4/4  Verify installation")
    print("=" * 70, flush=True)
    check = (
        "import torch, torchvision, torchaudio, transformers, peft;"
        "print('torch      :', torch.__version__);"
        "print('torchvision:', torchvision.__version__);"
        "print('torchaudio :', torchaudio.__version__);"
        "print('transformers:', transformers.__version__);"
        "print('peft       :', peft.__version__);"
        "ok = torch.cuda.is_available();"
        "print('CUDA available:', ok);"
        "print('GPU:', torch.cuda.get_device_name(0) if ok else 'NONE — cu128 install may have failed')"
    )
    subprocess.call([PY, "-c", check])

    print("\nDone. If CUDA available = True and every version ends in '+cu128',")
    print("you are ready. If CUDA is False, re-run this script.")


if __name__ == "__main__":
    main()

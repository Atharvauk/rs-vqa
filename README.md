# FM12A — Remote-Sensing Visual Question Answering (RS·VQA)

A self-contained project: a locally fine-tuned vision-language model that answers questions about
objects in satellite images, **shows the objects it detected**, and **abstains when unsure**.
Everything runs offline in 4-bit — no paid APIs in the answering path.

**Student:** Atharva Uday Kalase (40509608) · ECS8060 AI Engineering · Queen's University Belfast
**Best result:** 7B fine-tuned = **56.7% exact-match / 66.7% LLM-judge** (fixed 1,000-question VRSBench subset).

---

## Quick start (one command)

```bat
cd FM12A
python "RUN 0 - run_all.py"
```

The launcher walks you through everything: it runs library setup, checks the dataset, then lets you
choose which app to launch (always in 4-bit). Press **Ctrl+C** at any prompt to quit.

Prefer to run steps yourself? See "Manual setup / run" below.

---

## Folder structure

```
FM12A/
├── RUN 0 - run_all.py                 ← one-command launcher (setup → dataset check → launch)
├── RUN 1 - setup_libs.py              installs cu128 torch + all required libraries
├── RUN 2 - optional - download_vrsbench.py   download the dataset (~13 GB, run once)
├── RUN 3 - optional - unzip_vrsbench.py      extract the image archives
├── VRSBench_data/                     the dataset (fills after download; SHARED by all models)
│
├── final_dashboard.py                 ★ showcase demo — 7B fine-tuned, dark UI, coach tour,
│                                         answer + bounding boxes, About / Results tabs
├── compare_models.py                  before/after + 3B/7B comparison (same dark UI)
│
├── 3B/  7B/  API/                     model folders (all read ../VRSBench_data)
│   ├── step01..step14 .py, demo*.py, rescore*.py, api_*.py
│   ├── qlora_adapter_v2/  (3B)   |  qlora_adapter_7b/  (7B, best)   ← copy adapters here
│   └── results/
│
├── logo/                              brand pack — RSVQA icon + wordmark lockups (SVG + PNG)
├── how_it_works/                      step icons (line + colour) for the deck / report
│
├── FM-12a_Report_IEEE.docx            IEEE 4-page report
├── FM-12a_Final_Report.docx           full report (4-page body + appendix)
├── FM-12a_Presentation.pptx           detailed academic deck (13 slides)
├── FM-12a_Pitch_Deck.pptx             business pitch deck (8 slides)
├── FM-12a_Pitch.docx                  pitch script + demo playbook
├── Business Pitch.docx                short pitch script (< 3 min)
├── business pitch video.docx          video narration script (< 4 min) + showcase plan
├── new_viva.docx                      viva prep Q&A (latest results)
├── FM-12a_Intro.mp4 / _30s.mp4        branded intro animations
└── README.md
```

---

## The two demo apps

### `final_dashboard.py` — showcase demo (use this for the live demo / viva)
- Dark, branded UI with a gradient hero and **Demo / About / Results** tabs.
- Runs the **7B fine-tuned** model in 4-bit.
- A single **Answer** button returns the answer **and draws the detected-object bounding boxes together**.
- Interactive **step-by-step coach tour** (Upload → Ask → Answer) with a Next button and a congrats finish.
- Example-question chips; the detector runs on **CPU** so it never starves the 7B of GPU memory.

```bat
python final_dashboard.py --load-4bit          REM local:  http://127.0.0.1:7860
python final_dashboard.py --load-4bit --share  REM optional public link
python final_dashboard.py --mock               REM UI only, NO model (great for screenshots)
```

### `compare_models.py` — before/after comparison
- Pick **3B / 7B** and **After / Before / Compare both** to show the effect of fine-tuning.
- Keeps one model in GPU memory; toggles the LoRA adapter instantly (size switch reloads ~15–20 s).

```bat
python compare_models.py --load-4bit
```

---

## Manual setup / run

**First-time setup (once):**
```bat
cd FM12A
python "RUN 1 - setup_libs.py"                 REM uninstalls torch, installs cu128 build + all libs
python "RUN 2 - optional - download_vrsbench.py"
python "RUN 3 - optional - unzip_vrsbench.py"
```
Then **copy your adapters in**: `qlora_adapter_v2` → `3B/`, `qlora_adapter_7b` → `7B/`.

**Evaluate (from inside a model folder):**
```bat
cd 7B
python step06_zeroshot_vqa.py --limit 1000 --load-4bit --adapter qlora_adapter_7b --out preds_7b_finetuned.json
python rescore.py   --pred preds_7b_finetuned.json
python api_judge.py --pred preds_7b_finetuned.json
```
The API folder answers remotely (no GPU) — set the current tunnel URL in `api_test.py` / `api_judge.py`.

---

## Notes
- **Dataset is shared:** every model folder points to `../VRSBench_data`, so you download it once.
  The demo and compare apps do **not** need the dataset — only the evaluation scripts do.
- **Adapters** (`qlora_adapter_v2`, `qlora_adapter_7b`) are model weights — copy them into their folders.
- **VRAM (4-bit):** 3B ≈ 4.6 GB, 7B ≈ 6.5 GB — both fit the 8 GB laptop GPU. Always use `--load-4bit`.
- **Fine-tuning (QLoRA):** rank 8, α 16, targets q/k/v/o_proj, LR 2×10⁻⁵, answer-only loss masking,
  effective batch 8, 1,000 examples; 7B adapter ≈ 20 MB, ≈ 40–55 min to train.
- **Public link not working?** The `--share` tunnel is often blocked on university/corporate networks;
  use the local URL, or bind the LAN for same-network access.
- **Best result:** 7B fine-tuned = **56.7% exact-match / 66.7% LLM-judge**; it beats a general model
  4× its size — domain fit beats scale.

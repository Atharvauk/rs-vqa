# pipeline_7b — full pipeline on Qwen2.5-VL-7B

Same 14-step pipeline as the base project, but the answering model is
**Qwen/Qwen2.5-VL-7B-Instruct**. Use this to get the 7B point of the
3B → 7B → API scale comparison.

## Paths
- **Data is shared** with the base project (absolute path to
  `...\project\rsvqa_pipeline\VRSBench_data`) — nothing is duplicated.
- **Results are local** to this folder (`pipeline_7b\results\`), so 7B runs
  never overwrite your 3B results. `detections.json`, `baseline_majority.json`
  and `calibration.json` were copied in so the grounded / shift steps work.

## Run order (JupyterLab — open a file, set `NOTEBOOK_ARGS`, `%run` it)
Only the model steps differ from 3B; the rest are model-independent.

1. `step06_zeroshot_vqa.py`  → 7B zero-shot on 1,000 Q  → `results/preds_zeroshot.json`  (**the key comparison run, ~15–20 min**)
2. `step08_grounded_pipeline.py`  → 7B grounded (reuses seeded `detections.json`)  *(optional)*
3. `step13_qlora_finetune.py`  → 7B QLoRA fine-tune  *(optional — see VRAM note)*
4. `rescore.py` then `api_judge.py` on the new preds → final numbers.

## VRAM note (RTX 5060, 8 GB)
- **Zero-shot 7B (step06)** fits comfortably in 4-bit (~6.5 GB); `max_pixels`
  is already capped to 512×28×28 for headroom.
- **Fine-tuning 7B (step13)** is tight and may OOM. If it crashes, lower
  `--max-pixels` (e.g. `256*28*28`) and keep `--grad-accum` high. If it still
  won't fit, keep your fine-tune result on 3B and report 7B for inference only.

## Minimum useful run
Just run **step06** for the 7B zero-shot number — that alone gives you the
3B-vs-7B scale result.

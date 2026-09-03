"""
STEP 13 (7B) — QLoRA domain fine-tuning with the v2 fixes.
Fine-tunes Qwen2.5-VL-7B on a slice of VRSBench TRAIN ([vqa] items only).
 
v2 fixes vs the old script (which caused counting to collapse):
  * answer-only loss masking  (train on the ANSWER, not the prompt)
  * learning rate 2e-5        (was 1e-4)
  * VRAM-safe max_pixels 256*28*28 default (7B on 8 GB is tight)
  * saves to qlora_adapter_v2/
 
Evaluate afterwards:
    python step06_zeroshot_vqa.py --limit 1000 --load-4bit \
        --adapter qlora_adapter_v2 --out preds_finetuned_7b.json
DRY TEST (no GPU): python step13_qlora_finetune.py --dry-run
"""
import argparse, json, random, re, sys, time
from pathlib import Path
 
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "VRSBench_data"
IMG  = DATA / "Images_train" / "Images_train"
SYS  = ("You are answering questions about an overhead aerial/satellite image. "
        "Answer with a short word or phrase only, no explanation.")
 
# Used ONLY inside Jupyter (a notebook cell can't take CLI flags).
NOTEBOOK_ARGS = ["--train-n", "1000", "--epochs", "1", "--lr", "2e-5"]
 
 
def load_train_items(n, seed=0):
    """VRSBench train -> [{'image':.., 'question':.., 'answer':..}] ([vqa] only)."""
    raw = json.load(open(DATA / "VRSBench_train.json", encoding="utf-8"))
    items = []
    for r in raw:
        conv = r.get("conversations", [])
        if len(conv) < 2 or "[vqa]" not in conv[0]["value"]:
            continue
        q = re.sub(r"<image>\s*", "", conv[0]["value"]).replace("[vqa]", "").strip()
        items.append({"image": r["image"], "question": q,
                      "answer": conv[1]["value"].strip()})
    random.Random(seed).shuffle(items)
    items = [x for x in items if (IMG / x["image"]).exists()]
    return items[:n]
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--train-n", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)                 # v2: was 1e-4
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-pixels", type=int, default=256 * 28 * 28,  # v2: lower for 8 GB
                    help="cap vision tokens to control GPU memory")
    ap.add_argument("--out-dir", default="qlora_adapter_v2")          # v2: distinct name
    ap.add_argument("--dry-run", action="store_true")
    in_notebook = "ipykernel" in sys.modules
    args = ap.parse_args(NOTEBOOK_ARGS if in_notebook else None)
 
    items = load_train_items(args.train_n)
    print(f"Training examples ([vqa], train split only): {len(items)}")
    if not items:
        raise SystemExit("No usable train items — is Images_train unpacked?")
    print("Sample:", json.dumps(items[0], indent=1)[:300])
    if args.dry_run:
        print("\nDry-run OK — data loads and prompts build. Run on GPU for real.")
        return
 
    import torch
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2_5_VLForConditionalGeneration)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from qwen_vl_utils import process_vision_info
 
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, quantization_config=bnb, device_map={"": 0})
    proc = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
 
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
 
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=args.lr)
    print(f"Learning rate: {args.lr}  |  answer-only masking: ON  |  "
          f"max_pixels: {args.max_pixels}")
    model.train()
 
    step = 0
    t_start = time.perf_counter()
    total = len(items) * args.epochs
    done = 0
    for epoch in range(args.epochs):
        for i, x in enumerate(items, 1):
            messages = [
                {"role": "system", "content": SYS},
                {"role": "user", "content": [
                    {"type": "image", "image": str(IMG / x["image"])},
                    {"type": "text", "text": x["question"]}]},
                {"role": "assistant", "content": x["answer"]}]
 
            text = proc.apply_chat_template(messages, tokenize=False)
            imgs, vids = process_vision_info(messages)
            inputs = proc(text=[text], images=imgs, videos=vids,
                          return_tensors="pt").to(model.device)
 
            # --- answer-only masking: compute loss ONLY on the assistant answer ---
            prompt_msgs = messages[:-1]
            prompt_text = proc.apply_chat_template(prompt_msgs, tokenize=False,
                                                   add_generation_prompt=True)
            prompt_inputs = proc(text=[prompt_text], images=imgs, videos=vids,
                                 return_tensors="pt")
            prompt_len = prompt_inputs.input_ids.shape[1]
 
            labels = inputs.input_ids.clone()
            labels[labels == proc.tokenizer.pad_token_id] = -100
            labels[:, :prompt_len] = -100          # ignore the prompt tokens
 
            loss = model(**inputs, labels=labels).loss / args.grad_accum
            loss.backward()
            if i % args.grad_accum == 0:
                optim.step(); optim.zero_grad(); step += 1
 
            done += 1
            if done % 5 == 0:
                el = time.perf_counter() - t_start
                eta = el / done * (total - done)
                print(f"  epoch {epoch+1} item {i}/{len(items)} "
                      f"loss {loss.item()*args.grad_accum:.3f} | eta {eta/60:.1f} min",
                      flush=True)
 
    out = HERE / args.out_dir
    model.save_pretrained(out)
    proc.save_pretrained(out)
    print(f"\nAdapter saved to {out}  ({(time.perf_counter()-t_start)/60:.1f} min)")
    print("Evaluate with: python step06_zeroshot_vqa.py --limit 1000 --load-4bit "
          f"--adapter {args.out_dir} --out preds_finetuned_7b.json")
 
 
if __name__ == "__main__":
    main()
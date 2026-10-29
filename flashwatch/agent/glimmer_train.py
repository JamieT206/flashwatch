"""LoRA fine-tune of Muse Glimmer 30B (open weights) on Drought Monitor narratives, on Modal (DESIGN.md G2).

    pip install modal && modal setup                      # once, with your Modal account
    modal run flashwatch/agent/glimmer_train.py --smoke   # 2 steps on 16 weeks: checks the pipeline (~15 min)
    modal run flashwatch/agent/glimmer_train.py           # uploads data/glimmer/*.jsonl, trains, saves the adapter
    modal run flashwatch/agent/glimmer_train.py::score    # the adapter's drafts for the 12 validation weeks, scored

Needs a Modal secret named `huggingface` holding HF_TOKEN. The model (transformers 5.15+; a text and vision
model, of which only the language part is adapted) is cached on the Modal volume, so reruns skip the download.
Training pairs: data/glimmer/train.jsonl (749 weeks, 2005-2019) and val.jsonl (104 weeks, 2020-2021),
chat format (system, user = the week's numbers as JSON, assistant = the real narrative); ~2,200 tokens
median, 4,400 max. QLoRA: 4-bit base, LoRA rank 16 on the language model's attention and MLP projections,
2 epochs, loss on the narrative only (prompt-completion format). On one 80 GB GPU about an hour.
Every function runs on at most one GPU (max_containers=1) with a hard timeout. A vLLM web server was tried and
dropped (Oct 30): its engine needs a CUDA compiler the slim image lacks, and each request to the crashed server
started another GPU.
"""

from __future__ import annotations

from pathlib import Path

import modal

BASE = "meta-models/Muse-Glimmer-30B"
VOL = modal.Volume.from_name("flashwatch-glimmer", create_if_missing=True)
CACHE = {"HF_HOME": "/vol/hf"}                     # model weights kept on the volume between runs
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.15.0", "transformers==5.22.0", "peft==0.21.2", "bitsandbytes==0.50.2",
                      "accelerate==1.15.0", "datasets==5.1.0", "trl==1.17.0")
         .env(CACHE))
app = modal.App("flashwatch-glimmer", image=image)
TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"   # not the vision tower


@app.function(gpu="A100-80GB", timeout=4 * 3600, max_containers=1, volumes={"/vol": VOL},
              secrets=[modal.Secret.from_name("huggingface")])
def train(epochs: int = 2, rank: int = 16, lr: float = 2e-4, max_len: int = 4608, smoke: bool = False):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig, MuseGlimmerForConditionalGeneration
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(BASE)
    model = MuseGlimmerForConditionalGeneration.from_pretrained(
        BASE, device_map="auto", dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                                               llm_int8_skip_modules=["vision_tower", "vision_adapter",
                                                                      "vision_projection", "lm_head"]))
    ds = load_dataset("json", data_files={"train": "/vol/train.jsonl", "val": "/vol/val.jsonl"})
    # prompt = system + the week's numbers, completion = the real narrative: the loss is on the narrative only
    ds = ds.map(lambda r: {"prompt": r["messages"][:-1], "completion": r["messages"][-1:]},
                remove_columns=ds["train"].column_names)
    if smoke:
        ds["train"], ds["val"] = ds["train"].select(range(16)), ds["val"].select(range(4))
    cfg = SFTConfig(output_dir="/vol/run-smoke" if smoke else "/vol/run", num_train_epochs=epochs, learning_rate=lr,
                    per_device_train_batch_size=1, gradient_accumulation_steps=8, lr_scheduler_type="cosine",
                    warmup_steps=0.05, bf16=True, logging_steps=1 if smoke else 10, eval_strategy="epoch",
                    save_strategy="no" if smoke else "epoch", max_length=max_len, max_steps=2 if smoke else -1,
                    gradient_checkpointing=True, completion_only_loss=True, report_to=[])
    lora = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=TARGETS)
    tr = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["val"], peft_config=lora,
                    processing_class=tok)
    tr.train()
    if not smoke:
        tr.save_model("/vol/glimmer-lora")
    VOL.commit()
    return tr.state.log_history


@app.function(gpu="A100-80GB", timeout=1800, max_containers=1, volumes={"/vol": VOL},
              secrets=[modal.Secret.from_name("huggingface")])
def generate(chats: list, max_new_tokens: int = 1200, temperature: float = 0.2, batch: int = 6) -> list[str]:
    """Drafts from the base model in bf16 plus the saved adapter, for a list of chats (system + user)."""
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, MuseGlimmerForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    tok.pad_token = tok.pad_token or "<|finetune_right_pad|>"
    model = MuseGlimmerForConditionalGeneration.from_pretrained(BASE, device_map="auto", dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, "/vol/glimmer-lora").eval()
    out = []
    for i in range(0, len(chats), batch):
        enc = tok.apply_chat_template(chats[i:i + batch], add_generation_prompt=True, return_tensors="pt",
                                      padding=True, return_dict=True).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                               temperature=temperature, pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return out


@app.local_entrypoint()
def main(epochs: int = 2, smoke: bool = False):
    data = Path(__file__).resolve().parents[2] / "data" / "glimmer"   # on this PC, not in the container
    with VOL.batch_upload(force=True) as up:
        for name in ("train.jsonl", "val.jsonl"):
            up.put_file(str(data / name), f"/{name}")
    hist = train.remote(epochs=epochs, smoke=smoke)
    for h in hist:
        if "eval_loss" in h or "loss" in h:
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in h.items()})


@app.local_entrypoint()
def score(n: int = 12):
    """Scores the adapter like eval.glimmer_eval (same weeks and measures); writes eval_glimmer-lora.json."""
    import json
    from flashwatch.eval.glimmer_eval import pick, report
    from flashwatch.eval.glimmer_eval import score as score_draft
    rows = pick(n)
    drafts = generate.remote([r["messages"][:2] for r in rows])
    out = []
    for r, d in zip(rows, drafts):
        _, user, real = r["messages"]
        s = score_draft(d, real["content"], json.loads(user["content"]))
        out.append({"date": r["date"], **s, "draft": d, "real": real["content"]})
        print(f"{r['date']}: untraced {s['untraced']:.2f} states {s['states']:.2f} direction {s['direction']:.0f} "
              f"rouge-L {s['rouge_l']:.2f}")
    report("glimmer-lora", 0, out)

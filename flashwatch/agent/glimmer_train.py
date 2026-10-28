"""LoRA fine-tune of Muse Glimmer 30B (open weights) on Drought Monitor narratives, on Modal (DESIGN.md G2).

    pip install modal && modal setup                      # once, with your Modal account
    modal run flashwatch/agent/glimmer_train.py           # uploads data/glimmer/*.jsonl, trains, saves the adapter
    modal run flashwatch/agent/glimmer_train.py::serve    # vLLM with the adapter, OpenAI-compatible, for eval.glimmer_eval

Not yet run (written Oct 29 ahead of the Oct 30 training session); expect to adjust versions and names.
Training pairs: data/glimmer/train.jsonl (749 weeks, 2005-2019) and val.jsonl (104 weeks, 2020-2021),
chat format (system, user = the week's numbers as JSON, assistant = the real narrative); ~2,200 tokens
median, 4,400 max. QLoRA: 4-bit base, LoRA rank 16 on the attention and MLP projections, 2 epochs, loss on
the assistant turn only. On one 80 GB GPU this should take about an hour.
Then score with: META_API_BASE=<served url>/v1 META_MODEL=glimmer-lora python -m flashwatch.eval.glimmer_eval --n 40
"""

from __future__ import annotations

from pathlib import Path

import modal

BASE = "meta-models/Muse-Glimmer-30B"
VOL = modal.Volume.from_name("flashwatch-glimmer", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.4.1", "transformers==4.46.3", "peft==0.13.2", "bitsandbytes==0.44.1",
                      "accelerate==1.1.1", "datasets==3.1.0", "trl==0.12.1", "hf_transfer==0.1.8")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}))
app = modal.App("flashwatch-glimmer", image=image)
DATA = Path(__file__).resolve().parents[2] / "data" / "glimmer"


@app.function(gpu="A100-80GB", timeout=4 * 3600, volumes={"/vol": VOL}, secrets=[modal.Secret.from_name("huggingface")])
def train(epochs: int = 2, rank: int = 16, lr: float = 2e-4, max_len: int = 4608):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, device_map="auto", torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
    ds = load_dataset("json", data_files={"train": "/vol/train.jsonl", "val": "/vol/val.jsonl"})
    ds = ds.remove_columns([c for c in ds["train"].column_names if c != "messages"])
    cfg = SFTConfig(output_dir="/vol/run", num_train_epochs=epochs, learning_rate=lr, per_device_train_batch_size=1,
                    gradient_accumulation_steps=8, lr_scheduler_type="cosine", warmup_ratio=0.05, bf16=True,
                    logging_steps=10, eval_strategy="epoch", save_strategy="epoch", max_seq_length=max_len,
                    gradient_checkpointing=True, assistant_only_loss=True, report_to=[])
    lora = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    tr = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["val"], peft_config=lora,
                    processing_class=tok)
    tr.train()
    tr.save_model("/vol/glimmer-lora")
    VOL.commit()
    return tr.state.log_history


@app.function(gpu="A100-80GB", timeout=3600, volumes={"/vol": VOL}, secrets=[modal.Secret.from_name("huggingface")],
              image=image.pip_install("vllm==0.6.4"))
@modal.web_server(8000, startup_timeout=1200)
def serve():
    import subprocess
    subprocess.Popen(["python", "-m", "vllm.entrypoints.openai.api_server", "--model", BASE, "--enable-lora",
                      "--lora-modules", "glimmer-lora=/vol/glimmer-lora", "--max-model-len", "6144",
                      "--quantization", "bitsandbytes", "--load-format", "bitsandbytes", "--port", "8000"])


@app.local_entrypoint()
def main(epochs: int = 2):
    with VOL.batch_upload(force=True) as up:
        for name in ("train.jsonl", "val.jsonl"):
            up.put_file(str(DATA / name), f"/{name}")
    hist = train.remote(epochs=epochs)
    for h in hist:
        if "eval_loss" in h or "loss" in h:
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in h.items()})

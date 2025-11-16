### FILE: scripts/tune.py
"""
LoRA fine-tuning entrypoint.

Steps:
1. Ingest ../data to produce data/raw_corpus.txt and data/train.jsonl.
2. Tokenize prompt/completion pairs into short sequences.
3. Fine-tune a tiny causal LM with PEFT/LoRA using CPU-friendly defaults.
4. Save adapters + tokenizer to outputs/law-corpus-lora.

Usage:
    python scripts/tune.py
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

try:
    from build_jsonl_from_data import build_training_files, TRAIN_JSONL_PATH
except ImportError as exc:  # pragma: no cover - defensive
    raise SystemExit("Cannot import build_jsonl_from_data. Run from project root.") from exc


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_OUTPUT_DIR = PROJECT_ROOT / "outputs/law-corpus-lora"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a tiny LM using LoRA adapters.")
    parser.add_argument("--base_model", default="EleutherAI/gpt-neo-125M", help="Hugging Face model id.")
    parser.add_argument("--output_dir", default=str(ADAPTER_OUTPUT_DIR), help="Where to store the adapters/tokenizer.")
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="AdamW learning rate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size (keep tiny for CPU).")
    parser.add_argument("--max_length", type=int, default=512, help="Max tokens per example.")
    return parser.parse_args()


def ensure_training_files() -> Dict[str, int]:
    print("[info] Building corpus + JSONL from ../data ...")
    summary = build_training_files(force=True)
    if summary["jsonl_examples"] == 0:
        print("[error] Could not build any training data. Aborting.")
        sys.exit(1)
    return summary


def load_model_and_tokenizer(model_name: str):
    """Load base model, falling back to gpt2 on memory errors."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        active_model_name = model_name
    except Exception as exc:  # pragma: no cover - defensive
        if model_name == "gpt2":
            raise
        print(f"[warn] Failed to load {model_name}: {exc}. Falling back to gpt2.")
        active_model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(active_model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(active_model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token or "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    return model, tokenizer, active_model_name


def tokenize_dataset(train_path: Path, tokenizer, max_length: int):
    dataset = load_dataset("json", data_files=str(train_path))
    train_split = dataset["train"]

    def join_fields(example):
        prompt = (example.get("prompt") or "").strip()
        completion = (example.get("completion") or "").strip()
        merged = (prompt + "\n\n" + completion).strip()
        return {"text": merged}

    formatted = train_split.map(join_fields, remove_columns=train_split.column_names)
    formatted = formatted.filter(lambda x: bool(x["text"].strip()))

    if len(formatted) == 0:
        print("[error] Training dataset is empty after formatting.")
        sys.exit(1)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = formatted.map(tokenize, batched=True, remove_columns=["text"])
    return tokenized


def main():
    args = parse_args()
    summary = ensure_training_files()
    print(f"[info] Paragraphs: {summary['paragraphs']}, JSONL pairs: {summary['jsonl_examples']}")

    train_jsonl = TRAIN_JSONL_PATH
    if not train_jsonl.exists():
        print(f"[error] Missing training file at {train_jsonl}.")
        sys.exit(1)

    model, tokenizer, resolved_model_name = load_model_and_tokenizer(args.base_model)
    tokenized = tokenize_dataset(train_jsonl, tokenizer, args.max_length)

    print("[info] Applying LoRA adapters ...")
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=None,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_total_limit=1,
        save_strategy="no",
        report_to="none",
        fp16=False,
        bf16=False,
        disable_tqdm=False,
        gradient_accumulation_steps=1,
        max_steps=-1,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized, data_collator=data_collator)

    print(f"[info] Starting training on CPU using base model {resolved_model_name}.")
    trainer.train()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Saving adapters + tokenizer -> {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[info] All done. Use scripts/model.py for interactive Q&A.")


if __name__ == "__main__":
    main()

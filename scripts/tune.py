#!/usr/bin/env python3
"""
finetune_on_corpus.py
- Reads ../data (many file types) and builds a text dataset
- Fine-tunes a small causal LM using LoRA (PEFT) on that single corpus
- Saves adapter + tokenizer to outputs/law-corpus-lora
"""

import os
from pathlib import Path
import json
import pandas as pd
from pypdf import PdfReader
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

# ---------- CONFIG ----------
BASE_MODEL = "EleutherAI/gpt-neo-125M"   # change if you want another small model
OUTPUT_DIR = "outputs/law-corpus-lora"
TEMP_CORPUS = "data/raw_corpus.txt"     # created by this script
MAX_LENGTH = 512
BATCH_SIZE = 1
EPOCHS = 2          # set 1 for quick smoke test
LR = 5e-5
# ----------------------------

DATA_DIR = Path("../data")
os.makedirs("data", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_pdf(path):
    try:
        reader = PdfReader(path)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages).strip()
    except Exception as e:
        print("PDF read error", path, e)
        return ""

def collect_texts():
    texts = []
    for p in DATA_DIR.glob("**/*"):
        if p.is_dir(): continue
        suf = p.suffix.lower()
        try:
            if suf in (".txt", ".md"):
                texts.append(p.read_text(encoding="utf-8", errors="ignore"))
            elif suf == ".pdf":
                texts.append(read_pdf(str(p)))
            elif suf == ".csv":
                try:
                    df = pd.read_csv(p)
                    for _, row in df.iterrows():
                        texts.append(" ".join([str(v) for v in row.dropna().values]))
                except Exception as e:
                    print("CSV read failed:", p, e)
            elif suf == ".jsonl":
                # try to extract any textual fields per line
                for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if not line.strip(): continue
                    try:
                        obj = json.loads(line)
                        # heuristics: pick 'text' or join values
                        if isinstance(obj, dict):
                            if "text" in obj:
                                texts.append(obj["text"])
                            elif "content" in obj:
                                texts.append(obj["content"])
                            elif "prompt" in obj and "completion" in obj:
                                texts.append(obj["prompt"] + "\n" + obj["completion"])
                            else:
                                texts.append(" ".join([str(v) for v in obj.values()]))
                        else:
                            texts.append(str(obj))
                    except:
                        texts.append(line)
            elif suf == ".json":
                try:
                    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                    texts.append(json.dumps(data))
                except:
                    pass
            else:
                # skip unknown binary files
                pass
        except Exception as e:
            print("Error reading", p, e)
    # filter empties & short items
    texts = [t.strip().replace("\r", " ") for t in texts if isinstance(t, str) and len(t.strip()) > 50]
    return texts

def write_corpus(texts, out_path=TEMP_CORPUS):
    with open(out_path, "w", encoding="utf-8") as fh:
        for t in texts:
            # separate docs with blank line
            fh.write(t.replace("\n", " ") + "\n\n")

def main():
    print("Collecting texts from ../data ...")
    texts = collect_texts()
    print("Collected", len(texts), "documents.")
    if len(texts) == 0:
        raise SystemExit("No usable text found in ../data. Add some files and retry.")
    print("Writing corpus to", TEMP_CORPUS)
    write_corpus(texts)

    # Use datasets.load_dataset('text', ...) to create a dataset from the corpus
    ds = load_dataset("text", data_files={"train": TEMP_CORPUS})
    print("Dataset built:", ds)

    print("Loading tokenizer & model:", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    # ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer))

    def tokenize_function(examples):
        # chunk into examples by splitting on double newlines to keep doc boundaries
        texts = []
        for t in examples["text"]:
            # break long doc into chunks
            t = t.strip()
            if not t:
                continue
            # split into ~MAX_LENGTH-token chunks by rough character heuristic
            chunk_size = 2000
            for i in range(0, len(t), chunk_size):
                chunk = t[i:i+chunk_size]
                texts.append(chunk)
        return tokenizer(texts, truncation=True, max_length=MAX_LENGTH, return_special_tokens_mask=False)

    print("Tokenizing (this can take a little while)...")
    tokenized = ds["train"].map(tokenize_function, batched=True, remove_columns=["text"])

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=None  # let PEFT auto-detect; adjust if needed
    )
    print("Applying LoRA adapters...")
    model = get_peft_model(model, lora_config)

    # Training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_steps=50,
        save_total_limit=2,
        fp16=False,
        remove_unused_columns=False,
        report_to="none",
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized, data_collator=data_collator)

    print("Starting training (CPU will be slow).")
    trainer.train()
    print("Training done — saving model + tokenizer to", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("All done.")

if __name__ == "__main__":
    main()


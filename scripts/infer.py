#!/usr/bin/env python3
"""
qa_infer.py
Interactive question-answer loop using the fine-tuned adapter in outputs/law-corpus-lora
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "outputs/law-corpus-lora"   # where finetune_on_corpus.py saved model
BASE_MODEL = "EleutherAI/gpt-neo-125M"  # base model name used originally
MAX_NEW_TOKENS = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_dir=MODEL_DIR):
    print("Loading tokenizer & model from", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.to(DEVICE)
    return tokenizer, model

def make_prompt(question):
    # short template: ask the model to answer concisely using what it learned from the corpus
    template = (
        "You have been trained only on a legal corpus of documents. "
        "Answer the question concisely and cite (briefly) if you reference a clause or law.\n\n"
        "Question: {}\n\nAnswer:".format(question)
    )
    return template

def answer_question(tokenizer, model, question):
    prompt = make_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

def main():
    tokenizer, model = load_model()
    print("Interactive Q&A — type 'exit' or 'quit' to stop.")
    while True:
        q = input("\nYour question> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        ans = answer_question(tokenizer, model, q)
        print("\nModel answer:\n", ans)

if __name__ == "__main__":
    main()


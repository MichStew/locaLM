### FILE: scripts/model.py
"""
Interactive Q&A loop for the adapted legal corpus model.

Usage:
    python scripts/model.py

Optional flags let you point at a different adapter folder or tweak decoding.
Type 'exit' or Ctrl+D to leave the REPL.
"""

import argparse
from pathlib import Path

from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "outputs/law-corpus-lora"

PROMPT_TEMPLATE = (
    "You have been trained on a legal corpus. Answer concisely using only that knowledge.\n"
    "Question: {question}\n"
    "Answer:"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive adapter-powered Q&A REPL.")
    parser.add_argument("--adapter_dir", default=str(DEFAULT_ADAPTER_DIR), help="Directory containing LoRA adapters.")
    parser.add_argument("--base_model", default="EleutherAI/gpt-neo-125M", help="Base model id if adapters missing.")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum total tokens per prompt.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum tokens to generate per reply.")
    return parser.parse_args()


def load_chat_model(adapter_dir: Path, fallback_model: str):
    """Load tokenizer + (optionally) adapters from disk."""
    if adapter_dir.exists():
        peft_config = PeftConfig.from_pretrained(adapter_dir)
        base_model_name = peft_config.base_model_name_or_path or fallback_model
    else:
        base_model_name = fallback_model

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token or "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer))

    if adapter_dir.exists():
        print(f"[info] Loading adapters from {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir)
    else:
        print("[info] Adapter directory missing — using base model weights only.")

    model.eval()
    return model, tokenizer


def format_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip())


def chat_loop(model, tokenizer, max_length: int, max_new_tokens: int) -> None:
    print("Enter a legal question (blank line to skip, 'exit' to quit):")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[info] Exiting.")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("[info] Bye!")
            break

        prompt = format_prompt(question)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_length = inputs["input_ids"].shape[-1]
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_ids = outputs[0][input_length:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        print(f"{answer}\n")


def main():
    args = parse_args()
    adapter_dir = Path(args.adapter_dir)
    model, tokenizer = load_chat_model(adapter_dir, args.base_model)
    chat_loop(model, tokenizer, args.max_length, args.max_new_tokens)


if __name__ == "__main__":
    main()

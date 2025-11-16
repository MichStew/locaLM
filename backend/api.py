import logging
import os
from pathlib import Path
from typing import Tuple

from flask import Flask, jsonify, request
from flask_cors import CORS
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("law-corpus-backend")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = PROJECT_ROOT / "outputs/law-corpus-lora"
DEFAULT_BASE_MODEL = "EleutherAI/gpt-neo-125M"
PROMPT_TEMPLATE = (
    "You have been trained on a legal corpus. Answer the user in the most complete thoughts possible, give explanation and break ideas into sections of completeness."
)
GENERATION_KWARGS = {
    "do_sample": True,
    "temperature": 0.8,
    "top_p": 0.9,
    "repetition_penalty": 1.5,
    "no_repeat_ngram_size": 3,
    "max_new_tokens": 400,
}


def resolve_base_model(default_name: str) -> str:
    """Allow overriding the base model via LAW_CORPUS_BASE_MODEL env var."""
    override = os.environ.get("LAW_CORPUS_BASE_MODEL")
    if override:
        LOGGER.info("Using base model override from LAW_CORPUS_BASE_MODEL=%s", override)
        return override
    return default_name


def load_model_and_tokenizer() -> Tuple:
    if ADAPTER_DIR.exists():
        LOGGER.info("Loading adapters from %s", ADAPTER_DIR)
        peft_cfg = PeftConfig.from_pretrained(ADAPTER_DIR)
        base_name = peft_cfg.base_model_name_or_path or DEFAULT_BASE_MODEL
    else:
        LOGGER.warning("Adapter directory %s missing; using base model only.", ADAPTER_DIR)
        base_name = DEFAULT_BASE_MODEL

    base_name = resolve_base_model(base_name)
    tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token or "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer))

    if ADAPTER_DIR.exists():
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)

    model.eval()
    return model, tokenizer


MODEL, TOKENIZER = load_model_and_tokenizer()

app = Flask(__name__)
CORS(app)


def generate_answer(question: str) -> str:
    prompt = PROMPT_TEMPLATE.format(question=question.strip())
    inputs = TOKENIZER(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    input_length = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            **GENERATION_KWARGS,
            pad_token_id=TOKENIZER.pad_token_id,
            eos_token_id=TOKENIZER.eos_token_id,
        )
    answer_ids = outputs[0][input_length:]
    return TOKENIZER.decode(answer_ids, skip_special_tokens=True).strip()


@app.route("/api/ask", methods=["POST"])
@app.route("/ask", methods=["POST"])
def ask():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    if not question:
        return jsonify({"error": "Missing 'question' in request body."}), 400
    try:
        answer = generate_answer(question)
    except Exception as exc:  # pragma: no cover - runtime guard
        LOGGER.exception("Generation failed: %s", exc)
        return jsonify({"error": "Generation failed. Check backend logs."}), 500
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

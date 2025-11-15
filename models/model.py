import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import configuration.json from ../data

# -- Configure here --
model_name = "HuggingFaceTB/SmolLM3-3B"   # your desired model
fallback_model = "EleutherAI/gpt-neo-125M"  # smaller fallback that should run on CPU
max_new_tokens = 256
# ---------------------

# choose device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

def try_load(model_name, device):
    try:
        print(f"Loading tokenizer and model: {model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        print("Model loaded OK.")
        return tokenizer, model
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None, None

tokenizer, model = try_load(model_name, device)
if model is None:
    print(f"Attempting fallback model: {fallback_model}")
    tokenizer, model = try_load(fallback_model, device)
    if model is None:
        print("Fallback failed. You likely need to run on a machine with more RAM or use a smaller model.")
        sys.exit(1)

# Example prompt
prompt = "Give me a brief explanation of gravity in simple terms."
# your code used a chat template; if tokenizer implements apply_chat_template keep it, otherwise just use prompt
if hasattr(tokenizer, "apply_chat_template"):
    messages_think = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages_think, tokenize=False, add_generation_prompt=True)
else:
    text = prompt

# tokenize and send to device
model_inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=2048)
model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

# generate
with torch.no_grad():
    gen = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)

# decode relative to input length
generated_ids = gen[0]
input_len = model_inputs["input_ids"].shape[1]
output_ids = generated_ids[input_len:]
print("\n=== GENERATED ===\n")
print(tokenizer.decode(output_ids, skip_special_tokens=True))


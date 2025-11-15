# need to import data from seperate folder 
from transformers import AutoTokenizer, AutoModelForCausalLM, 

model_name = "HuggingFaceTB/SmolLM3-3B"
device ="cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained( 
model_name, 
).to(device)

# prepare the model input 
prompt = "you are an expert law professional"

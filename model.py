import logging
import torch
import os
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# Open config file
with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# Some logs for the errors
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=config['general']['logging_level'])

# Initializing
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = config['general']['pytorch_cuda_config']
torch.cuda.empty_cache()  # Clean the cache, recommended for low GPU's

# Obtaining some variables
hf_token = config['model']['token']
model_id = config['model']['id']

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, use_auth_token=hf_token)
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)

while True:
    torch.cuda.empty_cache()
    input_text = input("Query: ")
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids, max_length=200) # You can modify the length of the answer
    print(tokenizer.decode(outputs[0]))
import logging
import zmq
import torch
import os
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=config['general']['logging_level'])


def get_answer(input_text):

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids, max_length = 100)
    return tokenizer.decode(outputs[0])

# Initializing
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = config['general']['pytorch_cuda_config']
torch.cuda.empty_cache()

# Building Model
hf_token = config['model']['token']
model_id = config['model']['id']

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, use_auth_token=hf_token)
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)

# Building Socket 
context = zmq.Context()
socket = context.socket(zmq.REP)
print(f"Wating for connection...")
socket.bind(config['socket']['REP'])
print(f"[SUCCESS] Connection Stablished")

# Running
while True:

    input_text = socket.recv_string()

    print(f"Generating Output...")
    output = get_answer(input_text)
    print(f"Generation Finished") 

    torch.cuda.empty_cache()
    socket.send_string(output)


# How to Install Llama2 Locally
This guide will explain how to set up everything in Windows to run new Meta Llama2 70B model on your local computer without WebUI or WSL needed.

## Prerequisites

- CUDA capable computer (NVIDIA's graphics card).
- NVIDIA RTX 3070 or higher recommended (I'm using this one, and works right on the edge).
- 8GB VRAM (determined by the graphics card)
- 12GB RAM at least
- Some Command Line Skills & Patience

## Install Cuda Toolkit
Skip this step if already installed. This toolkit is necessary to harness the full potential of your computer. Trying to run Llama2 on CPU barely works. All the instalation guide can be found in this [CUDA Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html). However here is a summary of the process:
1. Check the compatibility of your NVIDIA graphics card with CUDA.
2. Update the drivers for your NVIDIA graphics card.
3. Download the CUDA Toolkit installer from the [NVIDIA official website](https://developer.nvidia.com/cuda-downloads).
4. Run the CUDA Toolkit installer.
5. Make sure the environment variables are set (specifically PATH).
6. Restart your computer.

Once it is installed in your computer verify the installation running `nvcc --version` in PowerShell. It should appear some info like this:

<p align="center">
  <img src="https://github.com/SamthinkGit/llama2-for-windows/assets/92941012/928d9b1b-e7b8-4c72-8986-07d0f98b5943" alt="NVIDIA CUDA installation" height=100/>
</p>


## Installing Dependencies
Ensure you have previously installed [Python](https://www.tutorialspoint.com/how-to-install-python-in-windows) and [pip](https://phoenixnap.com/kb/install-pip-windows). Then download all the necessary libraries using the terminal:

```powershell
# Base Dependencies
pip install transformers torch yaml
```

Llama2 isn't often used directly, so it is also necesary to integrate 4bit-optimization into the model. For this we must use [bitsandbytes](https://github.com/TimDettmers/bitsandbytes), however currently (v0.41.0) it has only CUDA support on Linux, so we will need to install a precompiled wheel in Windows. For this follow the next steps:
1. Check your CUDA version using `nvcc --version`
2. Download your wheel from [this repository](https://github.com/jllllll/bitsandbytes-windows-webui/releases/tag/wheels) (thanks to jllllll)
   **Note**: Must use at least >= 0.39.1 to work with 4bit-optimization
4. Install the library

```powershell
# Go to your wheel directory
cd (path-to-download)

# Replace with your selected wheel
pip install bitsandbytes-0.41.1-py3-none-win_amd64.whl

# Check it has been succesfully installed
pip show bitsandbytes

# Check if it has been compiled with CUDA support. Some versions can fail, however it is
# only necesary that the warning "This version has not been compiled with CUDA" DO NOT pop up
# (even if it crashes some lines afterwards there is no problem)
python -m bitsandbytes
```

Finally with bitsandbytes installed we will also add the accelerate library to optimice the model
```powershell
pip install accelerate
```

## Obtaining Access to Llama2
1. First, we need to create an accout into the [Hugging Face](https://huggingface.co/) page and get our **access token** to load the model in the computer. You can follow [this guide](https://huggingface.co/docs/hub/security-tokens#:~:text=To%20create%20an%20access%20token,clicking%20on%20the%20Manage%20button.) but is as simple as going to Settings > Access Tokens > New Token > Write.
<p align="center">
  <img src="https://github.com/SamthinkGit/llama2-for-windows/assets/92941012/58371a09-407a-4323-b74d-5178a412055c" alt="Hugging Face get a Key" height=400/>
</p>

2. With the **same email as the used in Hugging Face** we must request access to the model to Meta in [AI.meta.com](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).
<p align="center">
  <img src="https://github.com/SamthinkGit/llama2-for-windows/assets/92941012/17842fde-9157-4c89-998b-433856904c2b" alt="Hugging Face get a Key" height=300/>
</p>

3. Go to Hugging Face, log into your account and select one of the three Llama2 open source models. Then request access to them in [this link](https://huggingface.co/meta-llama/Llama-2-13b-hf). When your access has been granted (1-2h) you'll receive an email and also the site will update to be fully enabled. Then you can look at all the models in [HuggingFace meta-llama models](https://huggingface.co/meta-llama)
<p align="center">
  <img src="https://github.com/SamthinkGit/llama2-for-windows/assets/92941012/d7624fc0-dba4-4a05-a9c3-4145cea438d5" alt="Hugging Face get a Key" height=300/>
</p>

## Running Llama2
You now have everything you needed to run the Llama2 Model on your GPU. You can test this installation using the scripts added to this repository:
```powershell
# Get the repository and fill the data
git clone https://github.com/SamthinkGit/llama2-for-windows.git
cd .\llama2-for-windows
python.exe .\setup.py

# Llama2 only-terminal mode
python.exe .\llama2.py

# Llama2 Web based Interface
pip install zmq streamlit
python.exe .\llama2-web.py

# Don't forget to give a star to this repo if it worked for u :D
```

If you want to build your own code for Llama2 or more purpouses continue with the guide.

## Writing your first Q&A Llama2 Agent
We will now write our first code to make Llama2 talk and answer some questions. We wont go in much further detail since this is not a LLM course. The code adapted from [Haystack-Llama2-Guide](https://github.com/anakin87/llama2-haystack/blob/main/llama2-haystack.ipynb) from anakin87

1. First, create a new `config.yaml` file where we will write some information of the model.  The `<BATCH_SIZE>` will determine how your GPU uses your memory. I use a value of 25 and it's fine but it can be up to 100 or more if you are fine with it. The model selected is listed in the yaml, I recommend using the 7b one for the minimum computer requisites. The 13b will need around 12GB VRAM to work fine and 24GB for the 70b model. Remember to fill `<YOUR_HUGGING_FACE_TOKEN>` with the key obtained few steps ago.
```powershell
# -> powershell
New-Item config.yaml
notepad.exe config.yaml
```

```yaml
# -> config.yaml
# Possible llama models:
# Llama-2-7b-hf
# Llama-2-7b-chat-hf
# Llama-2-13b-hf
# Llama-2-13b-chat-hf
# Llama-2-70b-hf
# Llama-2-70b-chat-hf

general:
  logging_level: WARNING
  pytorch_cuda_config: max_split_size_mb:<BATCH_SIZE>

model:
  token: <YOUR_HUGGING_FACE_TOKEN>
  id: meta-llama/Llama-2-7b-chat-hf
```

2. Create a Python file `model.py` with your IDE or by using the notepad.exe in some dir. Then initialice the model settings written in the yaml.
```python
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
```

3. Finally we just need to build the model and write a query
```python
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, use_auth_token=hf_token)
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)

while True:
    torch.cuda.empty_cache()
    input_text = input("Query: ")
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids, max_length=100) # You can modify the length of the answer
    print(tokenizer.decode(outputs[0]))
```

### Results
```powershell
python.exe .\model.py
Loading checkpoint shards: 100%|█████████████████████████████████████| 2/2 [00:15<00:00,  7.52s/it]
Question: Explain briefly how to play Dark Souls I
[AI] <s> Explain briefly how to play Dark Souls I and II in multiplayer.
Hopefully this will help you get started with playing Dark Souls in multiplayer.
Dark Souls is a challenging action RPG with a unique sense of...
```

## Building a WebInterface for Llama2
We will now use streamlit and zmq to build a small server for our model to talk with the user. We can make it all in one script, however it is better to optimice it in two parallel executions releasing a bit the GPU from the loading of the web.

1. Add new socket parameters into the `config.yaml` file
```yaml
# Select the a free port, in this case 12443 is fine
socket:
  REQ: tcp://localhost:12443
  REP: tcp://*:12443
```

2. Create 2 new files, `model-back.py` and `model-front.py`. In the back we will copy the content of `model.py` and modify the code to accept queries from a server. In the front we will simply request a query to the user and print the generation in the screen.
```python
#model-back.py
import zmq
# Same imports...

# Initialize all variables (step 2 model.py)...

# Build the model
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, use_auth_token=hf_token)
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)

# Function to generate a output from a query
def get_answer(input_text):

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids, max_length = 100)
    return tokenizer.decode(outputs[0])

# Building Socket
context = zmq.Context()
socket = context.socket(zmq.REP)
print(f"Wating for connection...")
socket.bind(config['socket']['REP'])
print(f"[SUCCESS] Connection Stablished")

# New loop for receiving queries
while True:

    input_text = socket.recv_string()

    print(f"Generating Output...")
    output = get_answer(input_text)
    print(f"Generation Finished")

    torch.cuda.empty_cache()
    socket.send_string(output)
```

```python
# model-front.py
import streamlit as st
import zmq
import yaml

# Open yaml
with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# Build socket
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect(config['socket']['REQ'])

# Build interface
st.title('[ Talk with Meta-Llama2 ]')
user_input = st.text_input("Input:")

# Write the query
if st.button("Send"):
    socket.send_string(user_input)
    st.write(socket.recv_string())
```
3. Now you only need to launch this 2 scripts to have Llama2 fully working!
```powershell
python.exe model-back.py
Loading checkpoint shards: 100%|█████████████████████████████████████████████████| 2/2 [00:07<00:00,  3.75s/it]
Wating for connection...
[SUCCESS] Connection Stablished
...
```
```powershell
# In other terminal
streamlit run model-front.py

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.128:8501

```
<p align="center">
  <img src="https://github.com/SamthinkGit/llama2-for-windows/assets/92941012/4fc6dc1e-d043-4a77-a070-ec1e22da4c91" alt="Hugging Face get a Key" height=300/>
</p>

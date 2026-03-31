# MF2LLM: Complete Guide to Fine-Tuning LLMs with LLaMA-Factory on Ubuntu

**Target Models**: DeepSeek, Qwen, Llama, and other Hugging Face compatible models  
**Recommended Environment**: Ubuntu 20.04 / 22.04 (WSL or cloud servers like AutoDL, Alibaba Cloud, Tencent Cloud)

---

## 1. Environment Setup and Remote Connection

### 1.1 Install Required Tools
- Install **WSL Ubuntu** (Ubuntu 22.04 recommended)
- Install **Visual Studio Code**
- In VSCode, install the extension: **Remote - SSH**

### 1.2 Configure SSH Connection
Edit your local SSH config file `~/.ssh/config` and add the following:

```bash
Host my-ubuntu
    HostName 192.168.1.10        # Replace with your actual server IP
    User ubuntu
    Port 22

Connect to the server:
    Press Ctrl + Shift + P
    Type and select Remote-SSH: Connect to Host
    Choose my-ubuntu 
```

## 2. Create Conda Virtual Environment
    Bash# Configure conda to store packages and environments on the data disk
    mkdir -p /root/autodl-tmp/conda/pkgs
    conda config --add pkgs_dirs /root/autodl-tmp/conda/pkgs

    mkdir -p /root/autodl-tmp/conda/envs
    conda config --add envs_dirs /root/autodl-tmp/conda/envs

    # Create and activate the environment
    conda create -n llama-factory python=3.10 -y
    conda activate llama-factory

## 3. Install LLaMA-Factory
    Bash# Clone the repository
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory

    # Install with dependencies
    pip install -e ".[torch,metrics]"

    # Verify installation
    llamafactory-cli version

    # Launch Web UI
    llamafactory-cli webui
Port Forwarding: Set up SSH tunneling to access the Web UI from your local browser.
Reference: https://www.autodl.com/docs/ssh_proxy/

## 4. Download Base Model
    Bash# Create directory for base models
    mkdir -p /root/autodl-tmp/Hugging-Face

    # Set Hugging Face mirror and cache path
    export HF_ENDPOINT=https://hf-mirror.com
    export HF_HOME=/root/autodl-tmp/Hugging-Face

    # Make settings permanent
    echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
    echo 'export HF_HOME=/root/autodl-tmp/Hugging-Face' >> ~/.bashrc
    source ~/.bashrc

    # Verify
    echo $HF_ENDPOINT
    echo $HF_HOME

    # Install download tool
    pip install -U huggingface_hub

    # Download base model (supports resume)
    huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

## 5. Prepare Custom Dataset
### 5.1 Dataset Format Example (data/magic_conch.json)
    JSON[
    {
        "instruction": "Who are you?",
        "input": "",
        "output": "Hello, I am the Magic Conch of Krabby Patty King. I'm happy to serve you! I can answer any questions about Krabby Patty King and burger making."
    },
    {
        "instruction": "How to fix this error?",
        "input": "Error message: Burger recipe is empty.",
        "output": "According to the error, the recipe file may not be loaded correctly. Please try:\n1. Check if the recipe file exists and the path is correct.\n2. Reload or update the recipe file.\n3. Restart the machine if necessary."
    }
    ]
### 5.2 Register Dataset
    Edit data/dataset_info.json and add:
    JSON"magic_conch": {
    "file_name": "magic_conch.json"
    }

Place magic_conch.json into the LLaMA-Factory/data/ directory.

## 6. Fine-Tuning the Model
### 6.1 Web UI Configuration
    Fine-tuning Method: LoRA (recommended)
    Dataset: magic_conch
    Key Parameters:
        Parameter,Recommended Value,Description
        Learning Rate,5e-5 ~ 1e-4,Adjust based on loss curve
        Epochs,3 ~ 10,Number of training epochs
        Batch Size,1 ~ 4,Per device batch size
        Gradient Accumulation,4 ~ 16,Effective batch size
        Cutoff Length,2048 ~ 4096,Maximum sequence length
        Compute Type,bf16 / fp16,bf16 preferred
        Validation Set Proportion,0.1,Validation split
        Max Gradient Norm,1.0,Gradient clipping
### 6.2 Recommended Background Training
    nohup llamafactory-cli train \
        --model_name_or_path /root/autodl-tmp/Hugging-Face/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B \
        --dataset magic_conch \
        --template default \
        --finetuning_type lora \
        --lora_target all \
        --output_dir ./saves/magic_conch_lora \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-4 \
        --num_train_epochs 5 \
        --bf16 \
        > train.log 2>&1 &
Monitor loss in train.log. Adjust learning rate or epochs as needed.

## 7. Evaluation

    Test the model in the Web UI Chat / Prediction interface
    Compare performance before and after fine-tuning
    Review checkpoints (LoRA adapters contain trained A and B matrices)

    Optimization Tips if Results Are Suboptimal:

    Use a stronger base model
    Increase data quantity and quality
    Improve data cleaning and instruction design
    Tune hyperparameters (learning rate, LoRA rank, etc.)

## 8. Export and Merge the Model
    Bash# Create output directory
    mkdir -p ./merged_model

# Merge LoRA adapter with base model
    llamafactory-cli export \
    --model_name_or_path /root/autodl-tmp/Hugging-Face/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B \
    --adapter_name_or_path ./saves/magic_conch_lora/checkpoint-xxx \
    --template default \
    --finetuning_type lora \
    --export_dir ./merged_model/magic_conch_merged \
    --export_size 5 \
    --export_device cpu
    The merged model is ready for deployment with vLLM, Ollama, LMDeploy, etc.
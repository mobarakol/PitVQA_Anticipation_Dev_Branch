#!/bin/bash -l 
 
# Request a number of GPU cards, in this case 1 (the maximum is 2) 
#$ -l gpu=1
 
# Request wallclock time (format hours:minutes:seconds). 
#$ -l h_rt=48:0:0 

# tmem: GPU memory or RAM?
#$ -l tmem=24G

# use GPU
#$ -l gpu=true
#$ -l gpu_type=(rtx4090|rtx6000|rtx8000|a100)

# Set the name of the job. 
#$ -N vb-anti-img

# log file path
#$ -o /path/to/your/log-o
#$ -e /path/to/your/log-e

# Set the working directory 
#$ -wd /home/mobislam
 
# Path variables 
path_script_test="path/to/your/folder/main.py" 

# Activate the venv 
source /share/apps/source_files/python/python-3.8.5.source
source /share/apps/source_files/cuda/cuda-11.8.source
source /path/to/your/virtual_env/bin/activate

# Exporting CUDA Paths. cuDNN included in cuda paths. 
# Add the CUDA Path 
export PATH=/share/apps/cuda-11.8/bin:/usr/local/cuda-11.8/bin:${PATH} 
export LD_LIBRARY_PATH=/share/apps/cuda-11.8/lib64:/usr/local/cuda-11.8/lib:/lib64:${LD_LIBRARY_PATH} 
export CUDA_INC_DIR=/share/apps/cuda-11.8/include 
export LIBRARY_PATH=/share/apps/cuda-11.8/lib64:/usr/local/cuda-11.8/lib:/lib64:${LIBRARY_PATH}

# Hugging Face 缓存路径
export HF_HOME='/cluster/project7/Llava_2024/cache/huggingface/'
export HF_DATASETS_CACHE='/cluster/project7/Llava_2024/cache/huggingface/'

# PyTorch 缓存路径
export TORCH_HOME='/cluster/project7/Llava_2024/cache/torch'
export TORCH_EXTENSIONS_DIR='/cluster/project7/Llava_2024/cache/torch_extensions'

# Run commands with Python 
python3 ${path_script_test} \
--lr=0.0000002 \
--epochs=20 \
--question_len=50 \
--answer_len=52 \
--random_seed=42 \
--batch_size=32 \
--workers=4

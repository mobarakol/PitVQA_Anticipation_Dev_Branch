#!/bin/bash -l 
 
# Request a number of GPU cards, in this case 1 (the maximum is 2) 
#$ -l gpu=1
 
# Request wallclock time (format hours:minutes:seconds). 
#$ -l h_rt=36:0:0 

# tmem: GPU memory or RAM?
#$ -l tmem=16G

# use GPU
#$ -l gpu=true
#$ -l gpu_type=(rtx4090|rtx6000|rtx8000|a6000)

# Set the name of the job. 
#$ -N anti-infer

# log file path
#$ -o /cluster/project7/Endonasal_2024/log-o
#$ -e /cluster/project7/Endonasal_2024/log-e

# Set the working directory 
#$ -wd /home/mobislam
 
# Path variables 
path_script_test="/cluster/project7/Endonasal_2024/project_gen_11/3-pitvqa-anticipation/inference.py" 

# Activate the venv 
source /share/apps/source_files/python/python-3.8.5.source
source /share/apps/source_files/cuda/cuda-11.8.source
source /cluster/project7/Llava_2024/venvs/py38cu118/bin/activate

# Exporting CUDA Paths. cuDNN included in cuda paths. 
# Add the CUDA Path 
export PATH=/share/apps/cuda-11.8/bin:/usr/local/cuda-11.8/bin:${PATH} 
export LD_LIBRARY_PATH=/share/apps/cuda-11.8/lib64:/usr/local/cuda-11.8/lib:/lib64:${LD_LIBRARY_PATH} 
export CUDA_INC_DIR=/share/apps/cuda-11.8/include 
export LIBRARY_PATH=/share/apps/cuda-11.8/lib64:/usr/local/cuda-11.8/lib:/lib64:${LIBRARY_PATH}

# Hugging Face 缓存路径
export HF_HOME='/cluster/project7/Llava_2024/cache/huggingface/'
export HF_DATASETS_CACHE='/cluster/project7/Llava_2024/cache/huggingface/'
export TRANSFORMERS_CACHE='/cluster/project7/Llava_2024/cache/huggingface/transformers'

# PyTorch 缓存路径
export TORCH_HOME='/cluster/project7/Llava_2024/cache/torch'
export TORCH_EXTENSIONS_DIR='/cluster/project7/Llava_2024/cache/torch_extensions'

# Run commands with Python 
python3 ${path_script_test} 
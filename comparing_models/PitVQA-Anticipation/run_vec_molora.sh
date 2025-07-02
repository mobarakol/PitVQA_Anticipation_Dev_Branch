#!/bin/bash -l 
 
# Request a number of GPU cards, in this case 1 (the maximum is 2) 
#$ -l gpu=1
 
# Request wallclock time (format hours:minutes:seconds). 
#$ -l h_rt=48:0:0 

# tmem: GPU memory or RAM?
#$ -l tmem=24G

# use GPU
#$ -l gpu=true
#$ -l gpu_type=(rtx4090|rtx6000|a100|a6000)

# Set the name of the job. 
#$ -N vec-anti

# log file path
#$ -o /cluster/project7/Endonasal_2024/log-o
#$ -e /cluster/project7/Endonasal_2024/log-e

# Set the working directory 
#$ -wd /home/mobislam
 
# Path variables 
path_script_test="/cluster/project7/Endonasal_2024/project_gen_11/3-pitvqa-anticipation/main.py" 

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
python3 ${path_script_test} \
--lr=0.0000001 \
--epochs=10 \
--seq_length=100 \
--random_seed=42 \
--batch_size=32 \
--workers=4 \
--mora_base_rank=8 \
--mora_coeff 64 64 56 56 48 48 40 40 32 32 24 24 \
--lora_rank 32 32 28 28 24 24 20 20 16 16 12 12 \
--lora_alpha 32 32 28 28 24 24 20 20 16 16 12 12 \
--dropout=0.1 \
--checkpoint_dir='/cluster/project7/Endonasal_2024/project_gen_11/3-pitvqa-anticipation/cp/vec_mlr_anti_saved_weights'
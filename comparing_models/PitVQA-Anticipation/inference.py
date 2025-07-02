import os
import torch
import torch.utils.data
import numpy as np
import random

from torch import nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

import evaluate
from nltk.translate.bleu_score import corpus_bleu

from dataloader import PitVQAAnticipation
from model import PitVQAGen

import warnings
warnings.filterwarnings('ignore')


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def batch_greedy_search(images, questions, model, tokenizer, max_length, device):
    answers = []
    batch_size = len(questions)

    model.eval()
    with torch.no_grad():
        # prepare the prompts for the entire batch
        prompt_texts = [f"Question: {q}\nAnswer:" for q in questions]
        
        # Tokenize the prompts with padding to handle varying lengths
        prompt_inputs = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding='longest',
            add_special_tokens=False      
        )

        # prepare model inputs
        padded_input_ids = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
        padded_attention_mask = torch.zeros((batch_size, max_length), device=device)

        orig_length = prompt_inputs['input_ids'].size(1)
        padded_input_ids[:, :orig_length] = prompt_inputs['input_ids']
        padded_attention_mask[:, :orig_length] = prompt_inputs['attention_mask']

        images = images.to(device)

        # initialize tensors to store generated tokens
        only_answer_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        # track which sequences have finished generating 
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # record each sample length (number of non-eos tokens)
        valid_lengths = padded_attention_mask.sum(dim=1).long()
        batch_indices = torch.arange(batch_size, device=device)

        for _ in range(max_length):
            # forward pass through the model
            max_valid_lengths = max(valid_lengths).item()
            if max_valid_lengths == 65:
                break

            logits = model(
                image=images, 
                qa_inputs_ids=padded_input_ids[:, :max_valid_lengths], 
                qa_att_mask=padded_attention_mask[:, :max_valid_lengths]
            )

            # get next token
            last_valid_logits = logits[batch_indices, valid_lengths-1, :]
            next_token_ids = torch.argmax(last_valid_logits, dim=-1)

            # check EOS
            is_eos = (next_token_ids == tokenizer.eos_token_id)
            finished = finished | is_eos

            # insert next token at valid_lengths position
            padded_input_ids[batch_indices, valid_lengths] = next_token_ids
            padded_attention_mask[batch_indices, valid_lengths] = 1
            valid_lengths += 1

            # append the selected tokens to the generated_ids
            only_answer_ids = torch.cat([only_answer_ids, next_token_ids.unsqueeze(1)], dim=1) 

            # if all sequences have finished, exit early
            if finished.all():
                break

        # decode the generated tokens into strings
        generated_ids_cpu = only_answer_ids.cpu().tolist()  # Move to CPU and convert to list for processing
        for i in range(batch_size):
            try:  # Find the first occurrence of eos_token_id to truncate the answer
                eos_index = generated_ids_cpu[i].index(tokenizer.eos_token_id)
                answer_ids = generated_ids_cpu[i][:eos_index]
            except ValueError:  # If eos_token_id is not found, use all generated tokens
                answer_ids = generated_ids_cpu[i]
            
            # decode the token IDs to a string, skipping special tokens
            answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            answers.append(answer)
    return answers


def validate(args, val_loader, model, tokenizer, device):
    references = []
    hypotheses = []
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load('meteor')

    model.eval()
    with torch.no_grad():
        for i, (images, questions, answers) in enumerate(val_loader, 0):
            images = images.to(device)
            generated_answers = batch_greedy_search(
                images,
                questions,
                model,
                tokenizer,
                max_length=args.seq_length,
                device=device
            )

            # 将生成的答案和参考答案添加到列表中
            for ref, hyp in zip(answers, generated_answers):
                references.append([ref.split()]) 
                hypotheses.append(hyp.split()) 

        # 准备用于HuggingFace评估的格式
        ref_sentence = [' '.join(ref[0]) for ref in references]
        hyp_sentence = [' '.join(hyp) for hyp in hypotheses]
        
        # 使用HuggingFace的metrics计算分数
        results_bleu = bleu.compute(predictions=hyp_sentence, references=ref_sentence)
        results_rouge = rouge.compute(predictions=hyp_sentence, references=ref_sentence)
        results_meteor = meteor.compute(predictions=hyp_sentence, references=ref_sentence)
        
        print("HuggingFace Metrics Results:")
        print(f"Overall BLEU: {results_bleu['bleu']:.6f}")
        print(f"Rouge-1: {results_rouge['rouge1']:.6f}")
        print(f"Rouge-L: {results_rouge['rougeL']:.6f}")
        print(f"Meteor: {results_meteor['meteor']:.6f}")
        
        # 计算BLEU-1到BLEU-4分数
        metrics = {
            "BLEU-1": corpus_bleu(references, hypotheses, weights=(1.00, 0.00, 0.00, 0.00)),
            "BLEU-2": corpus_bleu(references, hypotheses, weights=(0.50, 0.50, 0.00, 0.00)),
            "BLEU-3": corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0.00)),
            "BLEU-4": corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
        }

        print("\nNLTK BLEU Scores:")
        for metric_name, score in metrics.items():
            print(f"{metric_name}: {score:.6f}")    


class HyperPara:
    def __init__(self):
        self.seq_length = 100
        self.workers = 8
        self.batch_size = 64


if __name__ == '__main__':

    random_seed = 42
    seed_everything(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = '/cluster/project7/Endonasal_2024/project_gen_11/3-pitvqa-anticipation/cp/'
    model_name = 'vec_mlr_anti_saved_weights_epoch_4_Best.pth.tar'
    model_path = save_dir + model_name
    print(f'model name: {model_path}')

    # load weights
    mora_base_rank = 8
    mora_coeff = [64, 64, 56, 56, 48, 48, 40, 40, 32, 32, 24, 24]
    lora_rank = [32, 32, 28, 28, 24, 24, 20, 20, 16, 16, 12, 12]
    lora_alpha = [32, 32, 28, 28, 24, 24, 20, 20, 16, 16, 12, 12]
    dropout = 0.1
    
    model = PitVQAGen(mora_base_rank=mora_base_rank, mora_rank_coefficients=mora_coeff,
                      lora_rank=lora_rank, lora_alpha=lora_alpha, dropout=dropout)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to(device)

    # prepare data
    args = HyperPara()
    val_seq = ['02', '06', '12', '13', '24']

    folder_head = r'/SAN/medic/surgicalLLM/content/PitVQA/datasets/PitVQA_Anticipation-25/QA/video_'
    folder_tail = '/*.txt'

    val_dataset = PitVQAAnticipation(val_seq, folder_head, folder_tail)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    validate(args, val_loader=val_dataloader, model=model, tokenizer=tokenizer, device=device)

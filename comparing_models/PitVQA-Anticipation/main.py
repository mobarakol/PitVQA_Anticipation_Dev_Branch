import os
import torch
import argparse
import torch.utils.data
import numpy as np
import random

from torch import nn
from utils import save_clf_checkpoint, adjust_learning_rate, AverageMeter
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from dataloader import PitVQAAnticipation
from model import PitVQAGen

import evaluate
from nltk.translate.bleu_score import corpus_bleu

import warnings
warnings.filterwarnings('ignore')


def train(args, train_dataloader, model, criterion, optimizer, epoch, tokenizer, device):
    model.train()
    total_loss = AverageMeter()

    for i, (images, questions, answers) in enumerate(train_dataloader, 0):
        # prepare prompts
        qa_prompt = [f'Question: {q}\nAnswer: {a}' for q, a in zip(questions, answers)]
        qa_prompt_inputs = tokenizer(qa_prompt, truncation=True, padding="max_length", max_length=int(args.seq_length), return_tensors="pt")
        
        # get labels
        labels = qa_prompt_inputs['input_ids'].clone()
        labels = labels.to(device)

        # for labels, mask question tokens and padding tokens
        for idx, q in enumerate(questions):
            q_prompt = f"Question: {q}\nAnswer: "
            q_length = len(tokenizer(q_prompt)["input_ids"]) - 1

            labels[idx, :q_length] = -100  # mask question
            eos_mask = (labels[idx] == tokenizer.eos_token_id)  # get all eos position
            if eos_mask.sum() > 1:  # if more than 1 eos 
                first_eos_pos = eos_mask.nonzero()[0].item()  # get first eos position
                labels[idx, (first_eos_pos+1):] = -100  # mask paddings, left one eos

        # get logits and labels
        logits = model(
                image=images.to(device), 
                qa_inputs_ids=qa_prompt_inputs['input_ids'].to(device),
                qa_att_mask=qa_prompt_inputs['attention_mask'].to(device)
        )

        # get shifted logits and labels
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # compute loss
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))  # (batch_size * seq_len, vocab_size)
        shift_labels = shift_labels.view(-1)  # (batch_size * seq_len) 
        loss = criterion(shift_logits, shift_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.update(loss.item())
    print("Training - Epoch: {}/{}, Loss: {:.6f}, AVG Loss: {:.6f}".format(epoch, args.epochs, total_loss.val, total_loss.avg))


def validate(args, val_loader, model, criterion, epoch, tokenizer, device):
    references = []
    hypotheses = []
    rouge = evaluate.load("rouge")
    meteor = evaluate.load('meteor')
    total_loss = AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, (images, questions, answers) in enumerate(val_loader, 0):
            # prepare prompts
            qa_prompt = [f'Question: {q}\nAnswer: {a}' for q, a in zip(questions, answers)]
            qa_prompt_inputs = tokenizer(qa_prompt, truncation=True, padding="max_length", max_length=int(args.seq_length), return_tensors="pt")
            
            # get labels
            labels = qa_prompt_inputs['input_ids'].clone()
            labels = labels.to(device)
            
            # for labels, mask question tokens and padding tokens
            answer_starts = []
            answer_ends = []
            for idx, q in enumerate(questions):
                q_prompt = f"Question: {q}\nAnswer: "
                q_length = len(tokenizer(q_prompt)["input_ids"]) - 1
                answer_starts.append(q_length+1)

                labels[idx, :q_length] = -100  # mask question
                eos_mask = (labels[idx] == tokenizer.eos_token_id)  # get all eos position
                if eos_mask.sum() > 1:  # if more than 1 eos 
                    first_eos_pos = eos_mask.nonzero()[0].item()  # get first eos position
                    labels[idx, (first_eos_pos+1):] = -100  # mask paddings, left one eos
                    answer_ends.append(first_eos_pos)

            # get logits and labels
            logits = model(
                image=images.to(device), 
                qa_inputs_ids=qa_prompt_inputs['input_ids'].to(device),
                qa_att_mask=qa_prompt_inputs['attention_mask'].to(device)
            )

            # get shifted logits and labels
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # compute loss
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))  # (batch_size * seq_len, vocab_size)
            shift_labels = shift_labels.view(-1)  # (batch_size * seq_len) 
            loss = criterion(shift_logits, shift_labels)
            total_loss.update(loss.item())

            # generate predicted answer
            _, predicted = torch.max(logits, dim=-1)
            # decode references and predictions
            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id  # tensor

            # extract answers
            reference_answers = []
            predicted_answers = []
            for idx, (start_pos, end_pos) in enumerate(zip(answer_starts, answer_ends)):
                # get answers 
                ref_tokens = labels_for_decode[idx, start_pos:end_pos]
                pred_tokens = predicted[idx, start_pos-1:end_pos-1]  # cause we are doing next token pred
                # decode
                ref = tokenizer.decode(ref_tokens, skip_special_tokens=True).strip()
                hyp = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
                # add to list
                reference_answers.append(ref)
                predicted_answers.append(hyp)

            # add answers to list
            for ref, hyp in zip(reference_answers, predicted_answers):
                references.append([ref.split()])
                hypotheses.append(hyp.split())

        ref_sentence = [' '.join(ref[0]) for ref in references] 
        hyp_sentence = [' '.join(hyp) for hyp in hypotheses]
        print("Eval - Epoch: {}/{}, Loss: {:.6f}, AVG Loss: {:.6f}".format(epoch, args.epochs, total_loss.val, total_loss.avg))

        # compute scores
        results_rouge = rouge.compute(predictions=hyp_sentence, references=ref_sentence)
        results_meteor = meteor.compute(predictions=hyp_sentence, references=ref_sentence)
        print("HF results: ")
        print(f"Rouge-1: {results_rouge['rouge1']:.6f}, Rouge-L: {results_rouge['rougeL']:.6f}, "
              f"Meteor: {results_meteor['meteor']:.6f}")

        # calculate BLEU-1~4
        metrics = {}
        metrics["Bleu_1"] = corpus_bleu(references, hypotheses, weights=(1.00, 0.00, 0.00, 0.00))
        metrics["Bleu_2"] = corpus_bleu(references, hypotheses, weights=(0.50, 0.50, 0.00, 0.00))
        metrics["Bleu_3"] = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0.00))
        metrics["Bleu_4"] = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
        print("NLTK results: ")
        print(f"BLEU-1: {metrics['Bleu_1']:.6f} BLEU-2: {metrics['Bleu_2']:.6f} "
              f"BLEU-3: {metrics['Bleu_3']:.6f} BLEU-4: {metrics['Bleu_4']:.6f}")
    return total_loss.avg


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_arg():
    parser = argparse.ArgumentParser(description='VisualQuestionAnswerGeneration')
    # Training parameters
    parser.add_argument('--epochs',         type=int,   default=60,   help='number of epochs to train for')
    parser.add_argument('--batch_size',     type=int,   default=32,   help='batch size')
    parser.add_argument('--workers',        type=int,   default=8,    help='for data-loading')
    parser.add_argument('--random_seed',    type=int,   default=42,   help='random seed')
    parser.add_argument('--seq_length',     type=int,   default=32,   help='sequence length for question and answer')

    parser.add_argument('--mora_base_rank', type=int, default=8, help='MoRA base rank')
    parser.add_argument('--mora_coeff', type=int, nargs='+', default=[32, 32, 30, 30, 28, 28, 26, 26, 24, 24, 22, 22], help='Vector-MoRA coefficient')

    parser.add_argument('--lora_rank', type=int, nargs='+', default=[16, 16, 14, 14, 12, 12, 10, 10, 8, 8, 6, 6], help='Vector-LoRA rank')
    parser.add_argument('--lora_alpha', type=int, nargs='+', default=[64, 64, 56, 56, 48, 48, 40, 40, 32, 32, 24, 24], help='Vector-LoRA alpha')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    parser.add_argument('--lr',             type=float, default=0.00002,  help='0.00001, 0.000005')
    parser.add_argument('--checkpoint_dir', default='./checkpoints/saved_weights_',  help='path to checkpoint')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_arg()
    seed_everything(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f'Batch size: {args.batch_size}')
    print(f'Learning rate: {args.lr}')
    print(f'Random seed: {args.random_seed}')
    print(f'MoRA base rank: {args.mora_base_rank}')
    print(f'Sequence length: {args.seq_length}')
    print(f'Dropout: {args.dropout}')

    start_epoch = 1
    epochs_since_improvement = 0
    best_val_loss = float('inf')

    # data location
    train_seq = ['01', '03', '04', '05', '07', '08', '09', '10', '11', '14',
                 '15', '16', '17', '18', '19', '20', '21', '22', '23', '25']
    val_seq = ['02', '06', '12', '13', '24']

    folder_head = r'/SAN/medic/surgicalLLM/content/PitVQA/datasets/PitVQA_Anticipation-25/QA_Anticipation/video_'
    folder_tail = '/*.txt'

    # dataloader
    train_dataset = PitVQAAnticipation(train_seq, folder_head, folder_tail)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.workers)
    val_dataset = PitVQAAnticipation(val_seq, folder_head, folder_tail)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers)

    # init tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    print(f'MoRA coeff: {args.mora_coeff}')
    print(f'LoRA ranks: {args.lora_rank}')
    print(f'LoRA alpha: {args.lora_alpha}')
    model = PitVQAGen(mora_base_rank=args.mora_base_rank, mora_rank_coefficients=args.mora_coeff,
                      lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, dropout=args.dropout)
    model = model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('model params: ', pytorch_total_params)

    # init optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100).to(device)

    # train and validation
    print('Start training.')
    for epoch in range(start_epoch, args.epochs+1):
        if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # train
        train(args, train_dataloader=train_dataloader, model=model, criterion=criterion, optimizer=optimizer,
              epoch=epoch, tokenizer=tokenizer, device=device)
        # validation
        val_loss = validate(args, val_loader=val_dataloader, model=model, criterion=criterion, 
                            epoch=epoch, tokenizer=tokenizer, device=device)

        if val_loss < best_val_loss:
            epochs_since_improvement = 0
            best_val_loss = val_loss
            save_dir = f'{args.checkpoint_dir}_epoch_{epoch}_'
            save_clf_checkpoint(save_dir, model)
            print('Best validation loss, model saved.')
        else:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
    print('End training.')

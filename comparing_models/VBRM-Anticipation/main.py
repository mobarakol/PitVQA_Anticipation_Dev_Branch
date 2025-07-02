import os
import torch
import argparse
import torch.utils.data
import numpy as np
import random

from torch import nn
from utils import save_clf_checkpoint, adjust_learning_rate, AverageMeter
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.nn.utils.rnn import pack_padded_sequence

from dataloader import PitVQAAnticipation
from model import VisualBertResMLPSentence

import evaluate
from nltk.translate.bleu_score import corpus_bleu

import warnings
warnings.filterwarnings('ignore')


def train(args, train_dataloader, model, criterion, optimizer, epoch, tokenizer, device):
    model.train()
    total_loss = AverageMeter()

    for i, (images, questions, answers) in enumerate(train_dataloader, 0):
        question_inputs = tokenizer(questions, padding="max_length", max_length=int(args.question_len),
                                    return_tensors="pt", truncation=True)
        answer_inputs = tokenizer(answers, padding="max_length", max_length=int(args.answer_len),
                                  return_tensors="pt", truncation=True)

        answers_GT_ID = answer_inputs.input_ids.to(device)
        answers_GT_len = torch.sum(answer_inputs.attention_mask, dim=1).unsqueeze(1).to(device)

        # get logits and labels
        scores, answer_GT_ID_sorted, decode_lengths, alphas, sort_ind = model(question_inputs.to(device), images.to(device), answers_GT_ID, answers_GT_len)

        adjusted_decode_lengths = [max(0, l-2) for l in decode_lengths]
        adjusted_decode_lengths = torch.tensor(adjusted_decode_lengths, dtype=torch.long)
        scores = pack_padded_sequence(scores[:, 2:], adjusted_decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(answer_GT_ID_sorted[:, 2:], adjusted_decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(scores, targets)

        # later integrate attention loss
        dec_alphas = alphas["dec_enc_attns"]
        alpha_trans_c = args.alpha_c / (args.n_heads * args.decoder_layers)
        for layer in range(args.decoder_layers):  # args.decoder_layers = len(dec_alphas)
            cur_layer_alphas = dec_alphas[layer]  # [batch_size, n_heads, 20, 26]
            for h in range(args.n_heads):
                cur_head_alpha = cur_layer_alphas[:, h, :, :]
                loss += alpha_trans_c * ((1. - cur_head_alpha.sum(dim=1)) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.update(loss.item(), sum(decode_lengths))
    print("Training - Epoch: {}/{}, Loss: {:.6f}, AVG Loss: {:.6f}".format(epoch, args.epochs, total_loss.val, total_loss.avg))


def validate(args, val_loader, model, criterion, epoch, tokenizer, device):
    references = []
    hypotheses = []
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load('meteor')

    model.eval()
    total_loss = AverageMeter()

    with torch.no_grad():
        for i, (images, questions, answers) in enumerate(val_loader, 0):
            question_inputs = tokenizer(questions, padding="max_length", max_length=int(args.question_len),
                                        return_tensors="pt", truncation=True)
            answer_inputs = tokenizer(answers, padding="max_length", max_length=int(args.answer_len),
                                      return_tensors="pt", truncation=True)

            answers_GT_ID = answer_inputs.input_ids.to(device)
            answers_GT_len = torch.sum(answer_inputs.attention_mask, dim=1).unsqueeze(1).to(device)

            # get logits and labels
            scores, answer_GT_ID_sorted, decode_lengths, alphas, sort_ind = model(question_inputs.to(device), images.to(device), answers_GT_ID, answers_GT_len)

            scores_copy = scores.clone()

            adjusted_decode_lengths = [max(0, l - 2) for l in decode_lengths]
            adjusted_decode_lengths = torch.tensor(adjusted_decode_lengths, dtype=torch.long)
            scores = pack_padded_sequence(scores[:, 2:], adjusted_decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(answer_GT_ID_sorted[:, 2:], adjusted_decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(scores, targets)

            # later integrate attention loss
            dec_alphas = alphas["dec_enc_attns"]
            alpha_trans_c = args.alpha_c / (args.n_heads * args.decoder_layers)
            for layer in range(args.decoder_layers):  # args.decoder_layers = len(dec_alphas)
                cur_layer_alphas = dec_alphas[layer]  # [batch_size, n_heads, 20, 196]
                for h in range(args.n_heads):
                    cur_head_alpha = cur_layer_alphas[:, h, :, :]
                    loss += alpha_trans_c * ((1. - cur_head_alpha.sum(dim=1)) ** 2).mean()

            total_loss.update(loss.item(), sum(decode_lengths))

            # references
            answer_GT_sorted = tokenizer.batch_decode(answer_GT_ID_sorted, skip_special_tokens=True)
            for answer_GT_sorted_i in answer_GT_sorted:
                references.append([answer_GT_sorted_i.split()])
            # hypotheses: predicted answer
            _, predicted_answer_id = torch.max(scores_copy, dim=2)
            predicted_answer = tokenizer.batch_decode(predicted_answer_id, skip_special_tokens=True)
            for pa in predicted_answer:
                hypotheses.append(pa.split())

        ref_sentence = [' '.join(ref[0]) for ref in references]
        hyp_sentence = [' '.join(hyp) for hyp in hypotheses]

        print(f"Epoch: {epoch}/{args.epochs} EVA LOSS: {total_loss.avg:.6f}")
        # compute
        results_bleu = bleu.compute(predictions=hyp_sentence, references=ref_sentence)
        results_rouge = rouge.compute(predictions=hyp_sentence, references=ref_sentence)
        results_meteor = meteor.compute(predictions=hyp_sentence, references=ref_sentence)
        print("HF results: ")
        print(f"BLEU-4: {results_bleu['bleu']:.6f}")
        print(f"Rouge1: {results_rouge['rouge1']:.6f}, RougeL: {results_rouge['rougeL']:.6f}, "
              f"Meteor: {results_meteor['meteor']:.6f}")

        # Calculate BLEU_1~4
        metrics = {}
        metrics["Bleu_1"] = corpus_bleu(references, hypotheses, weights=(1.00, 0.00, 0.00, 0.00))
        metrics["Bleu_2"] = corpus_bleu(references, hypotheses, weights=(0.50, 0.50, 0.00, 0.00))
        metrics["Bleu_3"] = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0.00))
        metrics["Bleu_4"] = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
        print("NLTK results: ")
        print(f"BLEU-1: {metrics['Bleu_1']:.6f} BLEU-2: {metrics['Bleu_2']:.6f} "
              f"BLEU-3: {metrics['Bleu_3']:.6f} BLEU-4: {metrics['Bleu_4']:.6f}")

    print("Eval - Epoch: {}/{}, Loss: {:.6f}, AVG Loss: {:.6f}".format(epoch, args.epochs, total_loss.val, total_loss.avg))
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

    parser.add_argument('--question_len',   type=int, default=32, help='sequence length for question')
    parser.add_argument('--answer_len',     type=int,   default=50,   help='sequence length for answer')

    parser.add_argument('--lr',             type=float, default=0.00002,  help='0.00001, 0.000005')
    parser.add_argument('--checkpoint_dir', default='./checkpoints/saved_weights_',  help='path to checkpoint')

    parser.add_argument('--alpha_c', type=float, default=1.0,
                        help='regularization parameter for doubly stochastic attention, as in the paper.')
    parser.add_argument('--emb_dim', type=int, default=300, help='dimension of word embeddings.')
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--encoder_layers', type=int, default=6, help='the number of layers of encoder in Transformer.')
    parser.add_argument('--decoder_layers', type=int, default=6, help='the number of layers of decoder in Transformer.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_arg()
    seed_everything(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    patch_size = 1  # 224=7
    train_dataset = PitVQAAnticipation(train_seq, folder_head, folder_tail)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_dataset = PitVQAAnticipation(val_seq, folder_head, folder_tail)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # init tokenizer and model
    token_size = int(args.question_len + (patch_size * patch_size))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = VisualBertResMLPSentence(vocab_size=len(tokenizer), embed_dim=args.emb_dim, encoder_layers=args.encoder_layers,
                                     decoder_layers=args.decoder_layers,
                                     dropout=args.dropout, n_heads=args.n_heads, token_size=token_size, answer_len=args.answer_len)
    model = model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('model params: ', pytorch_total_params)

    # init optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)

    # train and validation
    print('Start training.')
    for epoch in range(start_epoch, args.epochs+1):
        if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # train
        train(args, train_dataloader=train_dataloader, model=model, criterion=criterion, optimizer=optimizer,
              epoch=epoch, tokenizer=tokenizer, device=device)
        # validation
        val_loss = validate(args, val_loader=val_dataloader, model=model, criterion=criterion, epoch=epoch,
                           tokenizer=tokenizer, device=device)

        if val_loss < best_val_loss:
            epochs_since_improvement = 0
            best_val_loss = val_loss
            # save_dir = f'{args.checkpoint_dir}_epoch_{epoch}_'
            # save_clf_checkpoint(save_dir, epoch, model, optimizer)
            print('Best validation loss, model saved.')
        else:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
    print('End training.')

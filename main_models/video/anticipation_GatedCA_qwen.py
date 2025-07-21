import os
import torch
import torch.nn as nn
import argparse
import torch.utils.data
import numpy as np
import random
import warnings
import evaluate
import re

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import XCLIPModel, XCLIPProcessor
from peft import get_peft_model, TaskType, LoraConfig
from tqdm import tqdm

warnings.filterwarnings('ignore')

class VideoQADataset(Dataset):
    def __init__(self, image_root, qa_root, split='train', train_seq=None, val_seq=None, processor=None):
        self.image_root = image_root
        self.qa_root = qa_root
        self.sequence_length = 8
        self.processor = processor
        self.sequences = train_seq if split == 'train' else val_seq
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        for seq in self.sequences:
            folder = f'video_{seq.zfill(2)}'
            image_folder = os.path.join(self.image_root, folder)
            qa_folder = os.path.join(self.qa_root, folder)

            if not os.path.isdir(image_folder) or not os.path.isdir(qa_folder):
                continue

            frame_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
            frame_ids = sorted([int(f.split('.')[0]) for f in frame_files])

            for i in range(len(frame_ids) - self.sequence_length + 1):
                chunk = frame_ids[i:i+self.sequence_length]
                expected = list(range(chunk[0], chunk[0]+self.sequence_length))
                if chunk != expected:
                    continue  # Skip non-contiguous chunks

                frame_paths = [
                    os.path.join(folder, f"{fid:05d}.png") for fid in chunk
                ]

                last_frame_id = chunk[-1]
                qa_path = os.path.join(folder, f"{last_frame_id:05d}.txt")
                full_qa_path = os.path.join(self.qa_root, qa_path)

                if not os.path.isfile(full_qa_path):
                    continue

                samples.append((frame_paths, qa_path))
                
        print(f"Total video samples: {len(samples)}")
        return samples

    def _load_qa(self, qa_path):
        qa_pairs = []
        full_path = os.path.join(self.qa_root, qa_path)
        with open(full_path, 'r') as f:
            for line in f:
                if "|" in line:
                    q, a = line.strip().split("|", 1)
                    qa_pairs.append((q.strip(), a.strip()))
        return qa_pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, qa_path = self.samples[idx]
        frames = [Image.open(os.path.join(self.image_root, p)).convert('RGB') for p in frame_paths]
        # Use processor to process the batch of frames (videos expects [List[Image]])
        processed = self.processor(videos=[frames], return_tensors="pt")
        video_tensor = processed["pixel_values"].squeeze(0)  # [T, 3, 224, 224]
        qa_pairs = self._load_qa(qa_path)
        questions = [q for q, a in qa_pairs]
        answers = [a for q, a in qa_pairs]
        return video_tensor, questions, answers

def collate_qa_clipwise(batch):
    """Custom collate function to handle multiple QA pairs per video"""
    vids, flat_q, flat_a = [], [], []
    for video, qs, as_ in batch:
        for q, a in zip(qs, as_):
            vids.append(video)
            flat_q.append(q)
            flat_a.append(a)
    
    if len(vids) == 0:
        return torch.empty(0), [], []
        
    videos = torch.stack(vids, dim=0)   # (N, T, 3, 224, 224)
    return videos, flat_q, flat_a
    
    #####

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GatedCrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=8):
        super(GatedCrossAttentionFusion, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, vision_emb, text_emb):
        fused, _ = self.cross_attn(text_emb, vision_emb, vision_emb)
        gate_input = torch.cat([text_emb, fused], dim=-1)
        gate = self.gate(gate_input)
        out = gate * fused + (1 - gate) * text_emb
        return self.ln(out)


class PitVQAGen(nn.Module):
    def __init__(self, peft_config=None):
        super(PitVQAGen, self).__init__()

        # X-CLIP Model for proper video-text understanding with temporal features
        model_name = "microsoft/xclip-base-patch32"
        self.visual_encoder = XCLIPModel.from_pretrained(model_name)
        
        # Add projection layers to match expected fusion dimension
        self.video_proj = nn.Linear(512, 1024)  # X-CLIP video features are 512D, project to 2048D for fusion (Qwen 2.5-3B hidden size)
        
        # Cross-attention fusion module
        self.cross_attention_fusion = GatedCrossAttentionFusion(embed_dim=1024, num_heads=16)
        
        # Freeze vision encoder parameters
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
            
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
        self.tokenizer.pad_token = self.tokenizer.eos_token  # end of string

        # text encoder - use Qwen embedding for text features
        # self.text_encoder = self.visual_encoder.text_model  # REMOVE: do not use XCLIP text encoder
        
        # Qwen 2.5-3B decoder with LoRA
        qwen = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')
        self.qwen = get_peft_model(qwen, peft_config)
        self.qwen.print_trainable_parameters()  # Verify trainable LoRA parameters

    def forward(self, image, qa_inputs_ids, qa_att_mask):
        video = image.to(next(self.parameters()).device)
        video_embeds = self.visual_encoder.get_video_features(
            pixel_values=video,
            return_dict=True
        )  # [batch_size, 512]
        video_embeds = self.video_proj(video_embeds)  # [batch_size, 1024]
        # Repeat video_embeds across sequence length
        seq_len = qa_inputs_ids.shape[1]
        video_embeds = video_embeds.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, 1024]

        text_embeds = self.qwen.model.get_input_embeddings()(qa_inputs_ids)  # [batch_size, seq_len, 1024]

        fused_text_features = self.cross_attention_fusion(text_embeds, video_embeds)
        fused_embeds = fused_text_features  # [batch_size, seq_len, 1024]
        fused_att_mask = qa_att_mask  # [batch_size, seq_len]
        print("fused_embeds shape:", fused_embeds.shape)
        print("fused_att_mask shape:", fused_att_mask.shape)
        qwen_output = self.qwen(
            inputs_embeds=fused_embeds,
            attention_mask=fused_att_mask
        )
        print("qwen_output.logits shape:", qwen_output.logits.shape)
        return qwen_output.logits
    
    
    #####

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def adjust_learning_rate(optimizer, shrink_factor):
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

def train(args, train_dataloader, model, criterion, optimizer, epoch, tokenizer, device):
    model.train()
    total_loss = []

    for i, (images, questions, answers) in enumerate(train_dataloader, 0):
        if images.shape[0] == 0:
            continue  # Skip empty batch
        # prepare prompts
        qa_prompt = [f'Question: {q}\nAnswer: {a}' for q, a in zip(questions, answers)]
        qa_prompt_inputs = tokenizer(qa_prompt, truncation=True, padding="max_length", max_length=int(args.seq_length), return_tensors="pt")

        # get labels
        labels = qa_prompt_inputs['input_ids'].clone().to(device)
        print("qa_prompt_inputs['input_ids'].shape:", qa_prompt_inputs['input_ids'].shape)
        # mask question tokens and padding tokens, but keep answer tokens and first EOS
        for idx, q in enumerate(questions):
            q_prompt = f"Question: {q}\nAnswer: "
            q_length = len(tokenizer(q_prompt)["input_ids"]) - 1
            labels[idx, :q_length] = -100  # mask question
            # mask padding tokens
            pad_mask = (qa_prompt_inputs['attention_mask'][idx] == 0)
            labels[idx][pad_mask] = -100
            # keep only the first EOS, mask any after
            eos_mask = (labels[idx] == tokenizer.eos_token_id)
            eos_indices = torch.where(eos_mask)[0]
            if len(eos_indices) > 1:
                labels[idx, eos_indices[1]:] = -100

        # Check if all labels are -100 (would cause nan loss)


        logits = model(
            image=images,  # Model handles device transfer internally
            qa_inputs_ids=qa_prompt_inputs['input_ids'].to(device),
            qa_att_mask=qa_prompt_inputs['attention_mask'].to(device)
        )

        # get shifted logits and labels (standard autoregressive training)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        print(f"shift_logits shape: {shift_logits.shape}")  # [batch, seq_len-1, vocab_size]
        print(f"shift_labels shape: {shift_labels.shape}")  # [batch, seq_len-1]

        # Optionally, print how many non-masked labels remain
        print(f"Non-masked labels count: {(shift_labels != -100).sum().item()}")

        # compute loss only if batch is non-empty after shifting
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        print(f"shift_logits (flattened) shape: {shift_logits.shape}")  # [total_tokens, vocab_size]
        print(f"shift_labels (flattened) shape: {shift_labels.shape}")  # [total_tokens]

        loss = criterion(shift_logits, shift_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
    print("Training - Epoch: {}/{}, AVG Loss: {:.6f}".format(epoch, args.epochs, np.array(total_loss).mean()))


def validate(args, val_loader, model, criterion, epoch, tokenizer, device):
    total_loss = []
    model.eval()

    with torch.no_grad():
        for i, (images, questions, answers) in enumerate(val_loader, 0):
            if images.shape[0] == 0:
                continue  # Skip empty batch
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
                eos_mask = (labels[idx] == tokenizer.eos_token_id)  # get all EOS position
                if eos_mask.sum() > 1:  # if more than 1 EOS
                    first_eos_pos = eos_mask.nonzero()[0].item()  # get first EOS position
                    labels[idx, (first_eos_pos+1):] = -100  # mask paddings, left one EOS
                    answer_ends.append(first_eos_pos)

            # No need to add video token labels since we're using cross-attention fusion
            # labels remain the same size as text sequence

            # get logits and labels - no manual device transfer for images
            logits = model(
                image=images,  # Model handles device transfer internally
                qa_inputs_ids=qa_prompt_inputs['input_ids'].to(device),
                qa_att_mask=qa_prompt_inputs['attention_mask'].to(device)
            )

            # get shifted logits and labels (standard autoregressive training)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            

            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            loss = criterion(shift_logits, shift_labels)
            total_loss.append(loss.item())

    return np.array(total_loss).mean()


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
    parser.add_argument('--batch_size',     type=int,   default=2,   help='batch size')
    parser.add_argument('--workers',        type=int,   default=8,    help='for data-loading')
    parser.add_argument('--random_seed',    type=int,   default=42,   help='random seed')
    parser.add_argument('--seq_length',     type=int,   default=120,   help='sequence length for question and answer')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    parser.add_argument('--dataset',        default='endo',  help='endo / pit')
    parser.add_argument('--lr',             type=float, default=2e-5,  help='0.0000001, 0.00000005')
    parser.add_argument('--checkpoint_dir', default='Cross_Attention/',  help='path to checkpoint')
    parser.add_argument('--best_ckpt_name', default='anticipation_GatedCA_qwen.pth', help='best checkpoint filename')

    args = parser.parse_args([])
    return args


if __name__ == '__main__':

    args = get_arg()
    seed_everything(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Batch size: {args.batch_size}')
    print(f'Learning rate: {args.lr}')
    print(f'Random seed: {args.random_seed}')
    print(f'Sequence length: {args.seq_length}')
    print(f'Dropout: {args.dropout}')
    os.makedirs(args.checkpoint_dir, exist_ok = True)
    start_epoch = 1
    epochs_since_improvement = 0
    best_val_loss = float('inf')

    print(f'Dataset: {args.dataset}')
    train_dataloader = None
    val_dataloader = None

    # data location - PitVQA Anticipation dataset
    train_seq = ['01', '03', '04', '05', '07']
    # val_seq = ['02', '06', '12']
    #train_seq = ['01', '03', '04', '05', '07', '08', '09', '10', '11', '14','15', '16', '17', '18', '19', '20', '21', '22', '23', '25']
    val_seq = ['02', '06', '12','13', '24']    
    image_root = '/SAN/medic/surgicalLLM/content/PitVQA/datasets/PitVQA_Anticipation-25/images'
    qa_root = '/SAN/medic/surgicalLLM/content/PitVQA/datasets/PitVQA_Anticipation-25/QA_Anticipation'

    # Create X-CLIP processor for video processing
    processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")
    
    # dataloader
    train_dataset = VideoQADataset(
        image_root=image_root,
        qa_root=qa_root,
        split='train',
        train_seq=train_seq,
        val_seq=val_seq,
        processor=processor
    )
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.workers, pin_memory=True,
                                collate_fn=collate_qa_clipwise)
    val_dataset = VideoQADataset(
        image_root=image_root,
        qa_root=qa_root,
        split='val',
        train_seq=train_seq,
        val_seq=val_seq,
        processor=processor
    )
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers, pin_memory=True,
                                collate_fn=collate_qa_clipwise)
    
    print('training samples:', len(train_dataset), 'validation samples:', len(val_dataset))

    # init tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj"]
    )

    model = PitVQAGen(peft_config=lora_config)
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

        if val_loss < best_val_loss:  # save model with better validation loss
            epochs_since_improvement = 0
            best_val_loss = val_loss
            save_dir = f'{args.checkpoint_dir}/{args.best_ckpt_name}'
            torch.save(model.state_dict(), save_dir)
            model.tokenizer.save_pretrained(args.checkpoint_dir)
            print('Best validation loss, model saved.')
        else:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            
            # Early stopping after 5 consecutive epochs without improvement
            if epochs_since_improvement >= 5:
                print(f"\nEarly stopping triggered after {epochs_since_improvement} epochs without improvement.")
                print(f"Best validation loss: {best_val_loss:.6f}")
                break
                
    print('End training.')



#######  Inference ############################
print('Inferencing....')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_nlp_mettics(references, hypotheses):
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load('meteor')

    # compute HF metrics
    results_bleu = bleu.compute(predictions=hypotheses, references=references)
    results_rouge = rouge.compute(predictions=hypotheses, references=references)
    results_meteor = meteor.compute(predictions=hypotheses, references=references)

    return results_bleu, results_rouge, results_meteor

def extract_time(text):
    """
    Extracts all numeric time expressions from `text` (hours, minutes, seconds),
    converts each to minutes, and returns the total.
    If no unit is found on a match, treats it as minutes.
    Returns 0 if no numbers at all are found.
    """
    total_minutes = 0.0

    # 1) find all explicit matches with a unit
    #    e.g. "2 hours", "30 mins", "20 seconds"
    pattern = r"([-+]?[0-9]*\.?[0-9]+)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b"
    for val, unit in re.findall(pattern, text, re.IGNORECASE):
        value = float(val)
        unit_lower = unit.lower()
        
        if unit_lower in ['hours', 'hour', 'hrs', 'hr', 'h']:
            total_minutes += value * 60
        elif unit_lower in ['minutes', 'minute', 'mins', 'min', 'm']:
            total_minutes += value
        elif unit_lower in ['seconds', 'second', 'secs', 'sec', 's']:
            total_minutes += value / 60

    # 2) If we found at least one explicit-unit match, return the sum
    if total_minutes > 0:
        return total_minutes

    # 3) Otherwise, try to grab a lone number (treat as minutes)
    lone_num = re.search(r"([-+]?[0-9]*\.?[0-9]+)\b", text)
    if lone_num:
        return float(lone_num.group(1))
    else:
        return 0.0

def calculate_time_metrics(references, hypotheses):
    """
    Calculate MSE and MAE for time-related references and hypotheses.
    Only processes samples that contain time-related keywords (minutes, time).
    """
    # Filter for time-related samples
    time_refs = []
    time_hyps = []
    
    for ref, hyp in zip(references, hypotheses):
        if any(keyword in ref.lower() for keyword in ['minute', 'time', 'hour', 'second']):
            time_refs.append(ref)
            time_hyps.append(hyp)
    

    
    print(f"Found {len(time_refs)} time-related samples")
    
    # Extract numeric time values
    ref_times = [extract_time(ref) for ref in time_refs]
    hyp_times = [extract_time(hyp) for hyp in time_hyps]
    
    # Create pairs only when both time values were successfully extracted (non-zero)
    valid_pairs = [(r, h) for r, h in zip(ref_times, hyp_times) ]
    

    
    # Calculate MSE and MAE
    squared_errors = [((r - h) ** 2)/2 for r, h in valid_pairs]
    absolute_errors = [(abs(r - h))/2 for r, h in valid_pairs]
    
    mse = np.mean(squared_errors)
    mae = np.mean(absolute_errors)
    print(f"MSE: {mse}, MAE: {mae}")
    return mse, mae, len(valid_pairs)

def batch_greedy_search(images, questions, model, tokenizer, max_length, device):
    answers = []
    batch_size = len(questions)

    model.eval()
    with torch.no_grad():
        # Prepare the prompts for the entire batch
        prompt_texts = [f"Question: {q}\nAnswer:" for q in questions]

        # Tokenize the prompts with padding to handle varying lengths
        prompt_inputs = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding='longest',
            add_special_tokens=False
        )

        # Prepare model inputs (no video token needed with cross-attention)
        padded_input_ids = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
        padded_attention_mask = torch.zeros((batch_size, max_length), device=device)

        orig_length = prompt_inputs['input_ids'].size(1)
        padded_input_ids[:, :orig_length] = prompt_inputs['input_ids']
        padded_attention_mask[:, :orig_length] = prompt_inputs['attention_mask']

        # No manual device transfer for images - model handles it internally

        # Initialize tensors to store generated tokens
        only_answer_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)

        # Track which sequences have finished generating
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Record each sample length (number of non-eos tokens)
        valid_lengths = padded_attention_mask.sum(dim=1).long()
        batch_indices = torch.arange(batch_size, device=device)

        for _ in range(max_length):
            max_valid_lengths = max(valid_lengths).item()
            if max_valid_lengths >= max_length:
                break  # Stop if any sequence reached max_length

            # Forward pass through the model - no manual device transfer
            logits = model(image=images, qa_inputs_ids=padded_input_ids[:, :max_valid_lengths], qa_att_mask=padded_attention_mask[:, :max_valid_lengths])

            # Get next token probabilities from the correct position
            last_valid_logits = logits[batch_indices, valid_lengths - 1, :]  # -1 because no video token shift

            # Get next token
            next_token_ids = torch.argmax(last_valid_logits, dim=-1)

            # Check EOS
            is_eos = (next_token_ids == tokenizer.eos_token_id)
            finished = finished | is_eos  # Update finished status

            padded_input_ids[batch_indices, valid_lengths] = next_token_ids
            padded_attention_mask[batch_indices, valid_lengths] = 1
            valid_lengths += 1

            # Append the selected tokens to the generated_ids
            only_answer_ids = torch.cat([only_answer_ids, next_token_ids.unsqueeze(1)], dim=1)

            # If all sequences have finished, exit early
            if finished.all():
                break

        # Decode the generated tokens into strings
        generated_ids_cpu = only_answer_ids.cpu().tolist()  # Move to CPU and convert to list for processing
        for i in range(batch_size):
            # Find the first occurrence of eos_token_id to truncate the answer
            try:
                eos_index = generated_ids_cpu[i].index(tokenizer.eos_token_id)
                answer_ids = generated_ids_cpu[i][:eos_index]
            except ValueError:
                # If eos_token_id is not found, use all generated tokens
                answer_ids = generated_ids_cpu[i]

            # Decode the token IDs to a string, skipping special tokens
            answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            answers.append(answer)

    return answers

def inference(args, val_loader, model, tokenizer, device):
    references = []
    hypotheses = []

    model.eval()
    with torch.no_grad():
        for i, (images, questions, answers) in enumerate(tqdm(val_loader), 0):
            images = images.to(device)
            generated_answers = batch_greedy_search(
                images,
                questions,
                model,
                tokenizer,
                max_length=args.seq_length,
                device=device
            )

            references.extend(answers)
            hypotheses.extend(generated_answers)

    return references, hypotheses


lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    #target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    target_modules=["q_proj", "k_proj", "v_proj"]
)

model = PitVQAGen(peft_config=lora_config)
save_dir = f'{args.checkpoint_dir}/{args.best_ckpt_name}'
model.load_state_dict(torch.load(save_dir, map_location=device))
model.to(device)
model.eval()

# args.seq_length = 32

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
tokenizer.pad_token = tokenizer.eos_token

# Create X-CLIP processor for video processing
processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")

#args.batch_size = 200
val_dataset = VideoQADataset(
    image_root=image_root,
    qa_root=qa_root,
    split='val',
    train_seq=train_seq,
    val_seq=val_seq,
    processor=processor
)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size*2,
                            shuffle=False, num_workers=args.workers, pin_memory=True,
                            collate_fn=collate_qa_clipwise)


references, hypotheses  = inference(args, val_loader=val_dataloader, model=model, tokenizer=tokenizer, device=device)
mse, mae, valid_pairs_count = calculate_time_metrics(references, hypotheses)

# Save references and hypotheses to files (similar to endovis_llama_v4.py)
with open(f'{args.checkpoint_dir}/GatedCA_qwen_ref.txt', 'w') as f:
    for ref in references:
        f.write(ref + '\n')

with open(f'{args.checkpoint_dir}/GatedCA_qwen_hyp.txt', 'w') as f:
    for hyp in hypotheses:
        f.write(hyp + '\n')

print(f"References and hypotheses saved to {args.checkpoint_dir}")

results_bleu, results_rouge, results_meteor = get_nlp_mettics(references, hypotheses)

print(f"BLEU-1: {results_bleu['precisions'][0]:.6f}, "
      f"BLEU-2: {results_bleu['precisions'][1]:.6f}, "
      f"BLEU-3: {results_bleu['precisions'][2]:.6f}, "
      f"BLEU-4: {results_bleu['precisions'][3]:.6f}, "
      f"Overall BLEU: {results_bleu['bleu']:.6f}")

print(f"RougeL: {results_rouge['rougeL']:.6f}")
print(f"Meteor: {results_meteor['meteor']:.6f}")

print('First 5 Labels:')
print(references[:5])

print('First 5 Prediction:')
print(hypotheses[:5])

print("Inference completed.")
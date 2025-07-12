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
        
        # Use the last frame path as the representative image path for display
        representative_image_path = os.path.join(self.image_root, frame_paths[-1])
        
        return video_tensor, questions, answers, representative_image_path

def collate_qa_clipwise(batch):
    """Custom collate function to handle multiple QA pairs per video"""
    vids, flat_q, flat_a, flat_img_paths = [], [], [], []
    for video, qs, as_, img_path in batch:
        for q, a in zip(qs, as_):
            vids.append(video)
            flat_q.append(q)
            flat_a.append(a)
            flat_img_paths.append(img_path)
    
    if len(vids) == 0:
        return torch.empty(0), [], [], []
        
    videos = torch.stack(vids, dim=0)   # (N, T, 3, 224, 224)
    return videos, flat_q, flat_a, flat_img_paths
    
    #####

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, text_features, video_features):
        # text_features: [batch_size, seq_len, hidden_size]
        # video_features: [batch_size, 1, hidden_size]
        
        # Cross-attention: text attends to video
        attn_output, _ = self.multihead_attn(
            query=text_features,
            key=video_features,
            value=video_features
        )
        
        # Add & Norm
        text_features = self.norm1(text_features + self.dropout(attn_output))
        
        # Feed-forward
        ffn_output = self.ffn(text_features)
        text_features = self.norm2(text_features + self.dropout(ffn_output))
        
        return text_features


class PitVQAGen(nn.Module):
    def __init__(self, peft_config=None):
        super(PitVQAGen, self).__init__()

        # X-CLIP Model for proper video-text understanding with temporal features
        model_name = "microsoft/xclip-base-patch32"
        self.visual_encoder = XCLIPModel.from_pretrained(model_name)
        
        # Add projection layers to match expected fusion dimension
        self.video_proj = nn.Linear(512, 2048)  # X-CLIP video features are 512D, project to 2048D for fusion (Qwen 2.5-3B hidden size)
        self.text_proj = nn.Linear(512, 2048)   # X-CLIP text features are 512D, project to 2048D for fusion
        
        # Cross-attention fusion module
        self.cross_attention_fusion = CrossAttentionFusion(hidden_size=2048, num_heads=16, dropout=0.1)
        
        # Freeze vision encoder parameters
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
            
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B')
        self.tokenizer.pad_token = self.tokenizer.eos_token  # end of string

        # text encoder - use X-CLIP's text encoder for consistency
        # X-CLIP text encoder is accessible through visual_encoder.text_model
        self.text_encoder = self.visual_encoder.text_model
        
        # Update text encoder embeddings to match Qwen tokenizer vocabulary
        original_weights = self.text_encoder.embeddings.token_embedding.weight.data
        new_vocab_size = len(self.tokenizer)
        embedding_dim = self.text_encoder.embeddings.token_embedding.embedding_dim
        new_embeddings = nn.Embedding(new_vocab_size, embedding_dim)
        original_vocab_size = original_weights.shape[0]
        # Copy original weights and initialize new tokens randomly
        new_embeddings.weight.data[:original_vocab_size] = original_weights
        if new_vocab_size > original_vocab_size:
            # Initialize new tokens with small random values
            nn.init.normal_(new_embeddings.weight.data[original_vocab_size:], std=0.02)
        self.text_encoder.embeddings.token_embedding = new_embeddings

        # Qwen 2.5-3B decoder with LoRA
        qwen = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B')
        self.qwen = get_peft_model(qwen, peft_config)
        self.qwen.print_trainable_parameters()  # Verify trainable LoRA parameters

    def forward(self, image, qa_inputs_ids, qa_att_mask):
        # Process video with X-CLIP's proper temporal modeling
        # video shape: [batch_size, num_frames, 3, 224, 224]
        
        # Move video to device using proper device handling
        video = image.to(next(self.parameters()).device)

        # This method processes the entire video sequence with temporal modeling
        video_embeds = self.visual_encoder.get_video_features(
            pixel_values=video,
            return_dict=True
        )
        # video_embeds shape: [batch_size, 512] - properly aggregated video features with temporal info
        
        # Project to match expected dimension for fusion
        video_embeds = self.video_proj(video_embeds)  # [batch_size, 768]
        
        # Process text with X-CLIP's text encoder
        text_embeds = self.text_encoder(
            input_ids=qa_inputs_ids,
            attention_mask=qa_att_mask,
            return_dict=True
        )
        
        # Get text features from X-CLIP text encoder
        text_features = text_embeds.last_hidden_state  # [batch_size, seq_len, 512]
        
        # Project text features to match video features dimension
        text_features = self.text_proj(text_features)  # [batch_size, seq_len, 2048]
        
        # Fuse video and text features using cross-attention
        video_embeds = video_embeds.unsqueeze(1)  # [batch_size, 1, 2048]
        
        # Apply cross-attention fusion - text attends to video
        fused_text_features = self.cross_attention_fusion(text_features, video_embeds)
        
        # Use the fused text features directly (no concatenation bias)
        fused_embeds = fused_text_features  # [batch_size, seq_len, 2048]
        fused_att_mask = qa_att_mask  # [batch_size, seq_len]
        
        # Generate output using Qwen decoder
        qwen_output = self.qwen(
            inputs_embeds=fused_embeds,
            attention_mask=fused_att_mask
        )
        
        return qwen_output.logits
    
    def forward_xclip_fusion(self, image, qa_inputs_ids, qa_att_mask):
        """Alternative: Use X-CLIP's built-in multimodal fusion"""
        # Move video to device
        video = image.to(next(self.parameters()).device)
        
        # Use X-CLIP's multimodal processing directly
        outputs = self.visual_encoder(
            pixel_values=video,
            input_ids=qa_inputs_ids,
            attention_mask=qa_att_mask,
            return_dict=True
        )
        
        # Get the multimodal features that X-CLIP already computed
        # These are balanced video-text representations
        multimodal_embeds = outputs.text_embeds  # [batch_size, seq_len, 512]
        multimodal_embeds = self.text_proj(multimodal_embeds)  # [batch_size, seq_len, 2048]
        
        # Generate output using Qwen decoder
        qwen_output = self.qwen(
            inputs_embeds=multimodal_embeds,
            attention_mask=qa_att_mask
        )
        
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

    for i, (images, questions, answers, _) in enumerate(train_dataloader, 0):
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
            eos_mask = (labels[idx] == tokenizer.eos_token_id)  # get all EOS position
            if eos_mask.sum() > 1:  # if more than 1 EOS
                first_eos_pos = eos_mask.nonzero()[0].item()  # get first EOS position
                labels[idx, (first_eos_pos+1):] = -100  # mask paddings, left one EOS

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

        # compute loss
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
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
        for i, (images, questions, answers, _) in enumerate(val_loader, 0):
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

            # compute loss
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
    parser.add_argument('--seq_length',     type=int,   default=64,   help='sequence length for question and answer')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    parser.add_argument('--dataset',        default='endo',  help='endo / pit')
    parser.add_argument('--lr',             type=float, default=2e-5,  help='0.0000001, 0.00000005')
    parser.add_argument('--checkpoint_dir', default='checkpoints_llama/',  help='path to checkpoint')
    parser.add_argument('--best_ckpt_name', default='anticipation_Qwen_Video.pth', help='best checkpoint filename')

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
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B')
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        #target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        target_modules=["q_proj", "k_proj", "v_proj"]
    )

    model = PitVQAGen(peft_config=lora_config)
    model = model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('model params: ', pytorch_total_params)

    # init optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100).to(device)

    #train and validation
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

        if epochs_since_improvement >= 5:
            print(f"\nEarly stopping triggered! No improvement for {epochs_since_improvement} epochs.")
            print("Stopping training and proceeding to inference...")
            break
    print('End training.')



#######  Inference ############################
print('Inferencing....')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize args for inference section
args = get_arg()

# data location - PitVQA Anticipation dataset
# train_seq = ['01', '03', '04', '05', '07']
# val_seq = ['02', '06', '12']
image_root = '/SAN/medic/surgicalLLM/content/PitVQA/datasets/PitVQA_Anticipation-25/images'
qa_root = '/SAN/medic/surgicalLLM/content/PitVQA/datasets/PitVQA_Anticipation-25/QA_Anticipation'

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
    image_paths = []
    questions_list = []

    model.eval()
    with torch.no_grad():
        for i, (images, questions, answers, img_paths) in enumerate(tqdm(val_loader), 0):
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
            image_paths.extend(img_paths)
            questions_list.extend(questions)

    return references, hypotheses, image_paths, questions_list


lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = PitVQAGen(peft_config=lora_config)
save_dir = f'{args.checkpoint_dir}/{args.best_ckpt_name}'
model.load_state_dict(torch.load(save_dir, map_location=device))
model.to(device)
model.eval()

# args.seq_length = 32

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B')
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


references, hypotheses, image_paths, questions_list = inference(args, val_loader=val_dataloader, model=model, tokenizer=tokenizer, device=device)

# Save references and hypotheses to files (similar to endovis_llama_v4.py)
with open(f'{args.checkpoint_dir}/referencesVi_Qwen.txt', 'w') as f:
    for ref in references:
        f.write(ref + '\n')

with open(f'{args.checkpoint_dir}/hypothesesVi_Qwen.txt', 'w') as f:
    for hyp in hypotheses:
        f.write(hyp + '\n')

print(f"References and hypotheses saved to {args.checkpoint_dir}")

# Calculate time-related metrics
mse, mae, valid_pairs_count = calculate_time_metrics(references, hypotheses)

results_bleu, results_rouge, results_meteor = get_nlp_mettics(references, hypotheses)

print(f"BLEU-1: {results_bleu['precisions'][0]:.6f}, "
      f"BLEU-2: {results_bleu['precisions'][1]:.6f}, "
      f"BLEU-3: {results_bleu['precisions'][2]:.6f}, "
      f"BLEU-4: {results_bleu['precisions'][3]:.6f}, "
      f"Overall BLEU: {results_bleu['bleu']:.6f}")

print(f"Rouge1: {results_rouge['rouge1']:.6f}")
print(f"RougeL: {results_rouge['rougeL']:.6f}")
print(f"Meteor: {results_meteor['meteor']:.6f}")

# Display 7 random samples with their image information
print('\n' + '='*80)
print('3 RANDOM SAMPLES WITH IMAGE INFORMATION:')
print('='*80)

# Set random seed for consistent sample selection across different scripts
random.seed(42)
# Select 7 random indices
total_samples = len(references)
random_indices = random.sample(range(total_samples), min(7, total_samples))

for i, idx in enumerate(random_indices):
    print(f'\nSAMPLE {i+1} (Index: {idx}):')
    print('-' * 50)
    print(f'Question: {questions_list[idx]}')
    print(f'Reference: {references[idx]}')
    print(f'Hypothesis: {hypotheses[idx]}')
    print(f'Image path: {image_paths[idx]}')
    print('-' * 50)

print('First 5 Labels:')
print(references[:5])

print('First 5 Predictions:')
print(hypotheses[:5])

print("Inference completed.")

mse, mae, count = calculate_time_metrics(references, hypotheses)

if count > 0:
    print(f"Processed {count} time-related samples for metrics calculation.")
    print(f"Final MSE: {mse}, MAE: {mae}")
else:
    print("No time-related samples were processed for metrics calculation.")
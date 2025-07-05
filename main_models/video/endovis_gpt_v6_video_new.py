import os
import torch
import torch.nn as nn
import argparse
import torch.utils.data
import numpy as np
import random
import warnings
import evaluate

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import XCLIPModel, XCLIPProcessor
from peft import get_peft_model, TaskType, LoraConfig
from tqdm import tqdm

warnings.filterwarnings('ignore')

class EndoVis18VideoVQA(Dataset):
    def __init__(self, seq, folder_head, folder_tail, sequence_length=8, transform=None, processor=None):
        self.sequence_length = sequence_length
        self.folder_head = folder_head
        self.folder_tail = folder_tail
        self.sequences = seq
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        self.processor = processor  # X-CLIP processor for video processing
        
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        for curr_seq in self.sequences:
            seq_folder = os.path.join(self.folder_head, f"seq_{curr_seq}")
            left_fr_folder = os.path.join(seq_folder, 'left_fr')
            qa_folder = os.path.join(seq_folder, 'vqa', 'Sentence')
            
            if not os.path.isdir(left_fr_folder) or not os.path.isdir(qa_folder):
                continue

            frame_files = sorted([f for f in os.listdir(left_fr_folder) if f.endswith('.png')])
            if len(frame_files) == 0:
                continue
                
            frame_ids = sorted([int(f.replace('frame', '').replace('.png', '')) for f in frame_files])

            for i in range(len(frame_ids) - self.sequence_length + 1):
                chunk = frame_ids[i:i+self.sequence_length]
                expected = list(range(chunk[0], chunk[0]+self.sequence_length))
                if chunk != expected:
                    continue  # Skip non-contiguous chunks

                frame_paths = [
                    os.path.join(left_fr_folder, f"frame{fid:03d}.png") for fid in chunk
                ]

                last_frame_id = chunk[-1]
                qa_file = os.path.join(qa_folder, f"frame{last_frame_id:03d}_QA.txt")
                if not os.path.isfile(qa_file):
                    continue

                samples.append((frame_paths, qa_file))
                
        print(f"Total video samples: {len(samples)}")
        return samples

    def _load_qa(self, qa_file):
        qa_pairs = []
        try:
            with open(qa_file, "r") as f:
                lines = [line.strip("\n") for line in f if line.strip()]
            for line in lines:
                if "|" in line:
                    q, a = line.split("|", 1)
                    qa_pairs.append((q.strip(), a.strip()))
        except Exception as e:
            print(f"Error loading QA file {qa_file}: {e}")
        return qa_pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, qa_file = self.samples[idx]
        
        # Load frames as PIL Images
        frames = []
        for fp in frame_paths:
            try:
                frame = Image.open(fp).convert('RGB')
                frames.append(frame)
            except Exception as e:
                print(f"Error loading frame {fp}: {e}")
                frames.append(Image.new('RGB', (224, 224), color='black'))
        
        # Use X-CLIP processor to process video frames
        if self.processor:
            processed = self.processor(videos=[frames], return_tensors="pt")
            video_tensor = processed["pixel_values"].squeeze(0)  # Shape: [num_frames, 3, H, W]
        else:
            # Fallback to transform-based processing
            frames = [self.transform(f) for f in frames]
            video_tensor = torch.stack(frames)  # shape: [sequence_length, C, H, W]

        # Load QA pairs
        qa_pairs = self._load_qa(qa_file)
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
        self.video_proj = nn.Linear(512, 768)  # X-CLIP video features are 512D, project to 768D for fusion
        self.text_proj = nn.Linear(512, 768)   # X-CLIP text features are 512D, project to 768D for fusion
        
        # Cross-attention fusion module
        self.cross_attention_fusion = CrossAttentionFusion(hidden_size=768, num_heads=12, dropout=0.1)
        
        # Freeze vision encoder parameters
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
            
        # tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token  # end of string

        # text encoder - use X-CLIP's text encoder for consistency
        # X-CLIP text encoder is accessible through visual_encoder.text_model
        self.text_encoder = self.visual_encoder.text_model
        
        # Update text encoder embeddings to match GPT2 tokenizer vocabulary
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

        # GPT2 decoder with LoRA
        gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt = get_peft_model(gpt, peft_config)
        self.gpt.print_trainable_parameters()  # Verify trainable LoRA parameters

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
        text_features = self.text_proj(text_features)  # [batch_size, seq_len, 768]
        
        # Fuse video and text features using cross-attention
        video_embeds = video_embeds.unsqueeze(1)  # [batch_size, 1, 768]
        
        # Apply cross-attention fusion - text attends to video
        fused_text_features = self.cross_attention_fusion(text_features, video_embeds)
        
        # Use the fused text features directly (no concatenation bias)
        fused_embeds = fused_text_features  # [batch_size, seq_len, 768]
        fused_att_mask = qa_att_mask  # [batch_size, seq_len]
        
        # Generate output using GPT2 decoder
        gpt_output = self.gpt(
            inputs_embeds=fused_embeds,
            attention_mask=fused_att_mask
        )
        
        return gpt_output.logits
    
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
        multimodal_embeds = self.text_proj(multimodal_embeds)  # [batch_size, seq_len, 768]
        
        # Generate output using GPT2 decoder
        gpt_output = self.gpt(
            inputs_embeds=multimodal_embeds,
            attention_mask=qa_att_mask
        )
        
        return gpt_output.logits
    
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
    parser.add_argument('--epochs',         type=int,   default=80,   help='number of epochs to train for')
    parser.add_argument('--batch_size',     type=int,   default=2,   help='batch size')
    parser.add_argument('--workers',        type=int,   default=8,    help='for data-loading')
    parser.add_argument('--random_seed',    type=int,   default=42,   help='random seed')
    parser.add_argument('--seq_length',     type=int,   default=64,   help='sequence length for question and answer')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    parser.add_argument('--dataset',        default='endo',  help='endo / pit')
    parser.add_argument('--lr',             type=float, default=2e-5,  help='0.0000001, 0.00000005')
    parser.add_argument('--checkpoint_dir', default='checkpoints_llama/',  help='path to checkpoint')
    parser.add_argument('--best_ckpt_name', default='best_model_GPT_Video_new.pth', help='best checkpoint filename')

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

    # data location
    train_seq = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
    val_seq = [1, 5, 16]
    folder_head = '/SAN/medic/surgicalLLM/content/PitVQA/EndoVis-18-VQA/'
    folder_tail = '/vqa/Sentence/*.txt'

    # Create X-CLIP processor for video processing
    processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")
    
    # dataloader
    train_dataset = EndoVis18VideoVQA(train_seq, folder_head, folder_tail, sequence_length=8, processor=processor)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.workers, pin_memory=True,
                                collate_fn=collate_qa_clipwise)
    val_dataset = EndoVis18VideoVQA(val_seq, folder_head, folder_tail, sequence_length=8, processor=processor)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers, pin_memory=True,
                                collate_fn=collate_qa_clipwise)
    
    print('training samples:', len(train_dataset), 'validation samples:', len(val_dataset))

    # init tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"]
        #target_modules=["c_attn"]
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
    target_modules=["c_attn", "c_proj"]
    #target_modules=["c_attn"]
)

model = PitVQAGen(peft_config=lora_config)
save_dir = f'{args.checkpoint_dir}/{args.best_ckpt_name}'
model.load_state_dict(torch.load(save_dir, map_location=device))
model.to(device)
model.eval()

# args.seq_length = 32

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Create X-CLIP processor for video processing
processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")

#args.batch_size = 200
val_dataset = EndoVis18VideoVQA(val_seq, folder_head, folder_tail, sequence_length=8, processor=processor)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size*2,
                            shuffle=False, num_workers=args.workers, pin_memory=True,
                            collate_fn=collate_qa_clipwise)


references, hypotheses  = inference(args, val_loader=val_dataloader, model=model, tokenizer=tokenizer, device=device)

# Save references and hypotheses to files (similar to endovis_llama_v4.py)
with open(f'{args.checkpoint_dir}/referencesVi_GPT.txt', 'w') as f:
    for ref in references:
        f.write(ref + '\n')

with open(f'{args.checkpoint_dir}/hypothesesVi_GPT.txt', 'w') as f:
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
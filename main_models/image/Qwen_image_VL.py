import os
import glob
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import argparse
import evaluate
import re

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision.transforms.functional import InterpolationMode
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

os.environ['HF_HOME'] = '/SAN/medic/surgicalLLM/Qwen-model1'
hf_home = os.path.expanduser(
    os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
)
print(hf_home)


# point all caches into your models folder
# base_cache = "/SAN/medic/surgicalLLM/content/PitVQA/models"
# os.environ["HF_HOME"]            = f"{base_cache}/huggingface"
# os.environ["TRANSFORMERS_CACHE"] = f"{base_cache}/transformers"
# os.environ["TORCH_HOME"]         = f"{base_cache}/torch"

class Pit24VQAClassification(Dataset):
    def __init__(self, seq, folder_head, folder_tail):

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

        # files, question and answers
        filenames = []
        for curr_seq in seq:
            filenames = filenames + glob.glob(folder_head + str(curr_seq) + folder_tail)
        self.vqas = []
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines:
                answer = line.split('|')[1]
                if answer not in ['no_visible_instrument', 'no_secondary_instrument']:  # filter unknown answers
                    self.vqas.append([file, line])
        print('Total files: %d | Total question: %.d' % (len(filenames), len(self.vqas)))

        # Labels

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        qa_full_path = Path(self.vqas[idx][0])
        seq_path = qa_full_path.parents[2]
        video_num = qa_full_path.parts[-2]
        file_name = self.vqas[idx][0].split('/')[-1]

        # img
        #img_loc = os.path.join(seq_path, 'images', video_num, file_name.split('.')[0] + '.png')
        img_loc = os.path.join('/SAN/medic/CARES/mobarak/PitVQA/square_endo_images', video_num, file_name.split('.')[0] + '.png')

        raw_image = Image.open(img_loc).convert('RGB')
        img = self.transform(raw_image)

        # question and answer
        #question = self.vqas[idx][1].split('|')[0]
        #answer = self.vqas[idx][1].split('|')[1]
        #label = self.labels.index(str(answer))
        question, answer = self.vqas[idx][1].split('|') # splits qeustion and answer with |

        #return img_loc, img, question, label
        return img_loc, img, question, answer

import math
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class PitVQAGen(nn.Module):
    def __init__(self, peft_config=None):
        super(PitVQAGen, self).__init__()

        # Load Qwen2.5-VL-3B model with PEFT
        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        #processor = AutoProcessor.from_pretrained(model_id, force_download=True)
        self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
        )
        
        # Apply PEFT to the model
        if peft_config is not None:
            self.model = get_peft_model(self.base_model, peft_config)
            self.model.print_trainable_parameters()
        else:
            self.model = self.base_model

        # Load processor and tokenizer
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, image, qa_inputs_ids=None, qa_att_mask=None, messages=None):
        """
        Forward pass for Qwen2.5-VL model
        Args:
            image: PIL Images or tensor images
            qa_inputs_ids: input token ids
            qa_att_mask: attention mask
            messages: formatted messages for Qwen (alternative to qa_inputs_ids)
        """
        if messages is not None:
            # Use Qwen's native message format
            inputs = self.processor(
                text=[self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)],
                images=[image] if not isinstance(image, list) else image,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
            return self.model(**inputs)
        else:
            # Use traditional input_ids and attention_mask format
            inputs = {
                'input_ids': qa_inputs_ids,
                'attention_mask': qa_att_mask,
                'pixel_values': image.unsqueeze(0) if len(image.shape) == 3 else image  # Ensure batch dimension
            }
            inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
            return self.model(**inputs)

import os
import torch
import argparse
import torch.utils.data
import numpy as np
import random

from torch import nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

import evaluate
from nltk.translate.bleu_score import corpus_bleu
from peft import  TaskType, LoraConfig

import warnings
warnings.filterwarnings('ignore')

def adjust_learning_rate(optimizer, shrink_factor):
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

def train(args, train_dataloader, model, criterion, optimizer, epoch, tokenizer, device):
    model.train()
    total_loss = []

    for i, (img_paths, images, questions, answers) in enumerate(train_dataloader, 0):
        # Convert tensor images back to PIL Images for Qwen processor
        batch_pil_images = []
        for img_path in img_paths:
            pil_img = Image.open(img_path).convert('RGB')
            batch_pil_images.append(pil_img)
        
        # Prepare messages in Qwen format
        batch_messages = []
        for pil_img, q, a in zip(batch_pil_images, questions, answers):
            system_message = (
                "You are a surgical assistant AI for endonasal pituitary surgery. "
                "Rely on visual and textual input to deliver accurate, clinically relevant answers. "
                "Use proper surgical terminology. There are 3 phases, 15 steps, 18 instruments and 14 surgical activities. "
                "Time is measured in minutes. Only short sentence answers."
            )
            messages = [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": f"Question: {q}\nAnswer: {a}"}
                    ]
                }
            ]
            batch_messages.append(messages)

        # Process batch with Qwen processor
        batch_texts = []
        batch_images = []
        
        for messages in batch_messages:
            text = model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_texts.append(text)
            # Extract image from messages
            for content in messages[1]["content"]:
                if content["type"] == "image":
                    batch_images.append(content["image"])
                    break
        
        # Tokenize and prepare inputs
        inputs = model.processor(
            text=batch_texts,
            images=batch_images,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Prepare labels for loss calculation
        labels = inputs['input_ids'].clone()
        
        # Mask system message and question tokens (keep only answer tokens for loss)
        for idx, (q, a) in enumerate(zip(questions, answers)):
            # Find where the answer starts in the tokenized sequence
            answer_text = f"Answer: {a}"
            answer_tokens = tokenizer(answer_text, add_special_tokens=False)['input_ids']
            
            # Mask everything except answer tokens
            input_ids = inputs['input_ids'][idx]
            # Simple approach: mask first part, keep last part for answer
            answer_start = len(input_ids) - len(answer_tokens) - 10 # rough estimate
            labels[idx, :max(0, answer_start)] = -100
        
        # Forward pass
        outputs = model.model(**inputs, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
        
    print("Training - Epoch: {}/{}, AVG Loss: {:.6f}".format(epoch, args.epochs, np.array(total_loss).mean()))


def validate(args, val_loader, model, criterion, epoch, tokenizer, device):
    total_loss = []
    model.eval()
    with torch.no_grad():
        for i, (_, images, questions, answers) in enumerate(val_loader, 0):
            # Prepare messages in Qwen format (similar to training)
            batch_messages = []
            for img, q, a in zip(images, questions, answers):
                system_message = (
                    "You are a surgical assistant AI for endonasal pituitary surgery. "
                    "Rely on visual and textual input to deliver accurate, clinically relevant answers. "
                    "Use proper surgical terminology. There are 3 phases, 15 steps, 18 instruments and 14 surgical activities. "
                    "Time is measured in minutes. Only short sentence answers."
                )
                messages = [
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": f"Question: {q}\nAnswer: {a}"}
                        ]
                    }
                ]
                batch_messages.append(messages)

            # Process batch with Qwen processor
            batch_texts = []
            batch_images = []
            
            for messages in batch_messages:
                text = model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                batch_texts.append(text)
                # Extract image from messages
                for content in messages[1]["content"]:
                    if content["type"] == "image":
                        batch_images.append(content["image"])
                        break
            
            # Tokenize and prepare inputs
            inputs = model.processor(
                text=batch_texts,
                images=batch_images,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Prepare labels for loss calculation
            labels = inputs['input_ids'].clone()
            
            # Mask system message and question tokens (keep only answer tokens for loss)
            for idx, (q, a) in enumerate(zip(questions, answers)):
                answer_text = f"Answer: {a}"
                answer_tokens = tokenizer(answer_text, add_special_tokens=False)['input_ids']
                answer_start = len(inputs['input_ids'][idx]) - len(answer_tokens) - 10
                labels[idx, :max(0, answer_start)] = -100
            
            # Forward pass
            outputs = model.model(**inputs, labels=labels)
            loss = outputs.loss
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
    parser.add_argument('--batch_size',     type=int,   default=4,   help='batch size (reduced for Qwen2.5-VL)')
    parser.add_argument('--workers',        type=int,   default=8,    help='for data-loading')
    parser.add_argument('--random_seed',    type=int,   default=42,   help='random seed')
    parser.add_argument('--seq_length',     type=int,   default=117,   help='sequence length for question and answer')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    parser.add_argument('--dataset',        default='endo',  help='endo / pit')
    parser.add_argument('--lr',             type=float, default=2e-5,  help='0.0000001, 0.00000005')
    parser.add_argument('--checkpoint_dir', default='Anticipation_checkpoints/',  help='path to checkpoint')
    
    # Checkpoint related arguments
    parser.add_argument('--best_ckpt_name', default='qwen_image.pth', help='name of the best checkpoint file')
    parser.add_argument('--resume_training', action='store_true', help='resume training from checkpoint')
    parser.add_argument('--save_every', type=int, default=10, help='save periodic checkpoint every N epochs')

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
    print(f'Checkpoint directory: {args.checkpoint_dir}')
    print(f'Resume training: {args.resume_training}')
    print(f'Save checkpoint every: {args.save_every} epochs')
    os.makedirs(args.checkpoint_dir, exist_ok = True)
    start_epoch = 1
    epochs_since_improvement = 0
    best_val_loss = float('inf')

    #print(f'Dataset: {args.dataset}')
    train_dataloader = None
    val_dataloader = None

    # data location
    #train_seq = ['01', '03', '04', '05', '07', '08', '09', '10', '11', '14','15', '16', '17', '18', '19', '20', '21', '22', '23', '25']
    train_seq = ['01', '03', '04', '05', '07']
    val_seq = ['02', '06', '12','13', '24']
    #train_seq = ['06']
    #val_seq = ['02']
    folder_head = '/SAN/medic/surgicalLLM/content/PitVQA/datasets/PitVQA_Anticipation-25/QA_Anticipation/video_'
    folder_tail = '/*.txt'

    # dataloader
    train_dataset = Pit24VQAClassification(train_seq, folder_head, folder_tail)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers)
    val_dataset = Pit24VQAClassification(val_seq, folder_head, folder_tail)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size*3,
                                shuffle=False, num_workers=args.workers)

    # init tokenizer and model
    # Note: The model's own tokenizer and processor will be used
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj"]  # Qwen-specific modules
    )

    model = PitVQAGen(peft_config=lora_config)
    model = model.to(device)

    # Use the model's tokenizer
    tokenizer = model.tokenizer

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('model params: ', pytorch_total_params)

    # init optimizer (no need for separate criterion as Qwen handles loss internally)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = None  # Qwen model handles loss computation

    # Try to load checkpoint and resume training
    checkpoint_path = f'{args.checkpoint_dir}/{args.best_ckpt_name}'
    if os.path.exists(checkpoint_path) and args.resume_training:
        try:
            print("Loading checkpoint...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                if 'best_val_loss' in checkpoint:
                    best_val_loss = checkpoint['best_val_loss']
                if 'epochs_since_improvement' in checkpoint:
                    epochs_since_improvement = checkpoint['epochs_since_improvement']
            else:
                # Legacy format - direct state dict
                model.load_state_dict(checkpoint)
            
            print(f"Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.6f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch...")
            start_epoch = 1
            best_val_loss = float('inf')
            epochs_since_improvement = 0

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
            
            # Save comprehensive checkpoint
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'epochs_since_improvement': epochs_since_improvement,
                'args': vars(args)
            }
            
            torch.save(checkpoint_data, save_dir)
            model.tokenizer.save_pretrained(args.checkpoint_dir)
            print(f'Best validation loss: {best_val_loss:.6f}, model saved.')
        else:
            epochs_since_improvement += 1
            print(f"\nEpochs since last improvement: {epochs_since_improvement}\n")

        if epochs_since_improvement >= 5:
            break
        # Save periodic checkpoint every N epochs
        if epoch % args.save_every == 0:
            periodic_save_dir = f'{args.checkpoint_dir}/checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'epochs_since_improvement': epochs_since_improvement,
                'args': vars(args)
            }, periodic_save_dir)
            print(f'Periodic checkpoint saved at epoch {epoch}')
    print('End training.')


from tqdm import tqdm
import evaluate

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import ViTModel, BlipTextModel
from peft import get_peft_model
from peft import  TaskType, LoraConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load('meteor')
def get_nlp_mettics(references, hypotheses):


    # compute HF metrics
    results_bleu = bleu.compute(predictions=hypotheses, references=references)
    results_rouge = rouge.compute(predictions=hypotheses, references=references)
    results_meteor = meteor.compute(predictions=hypotheses, references=references)

    return results_bleu, results_rouge, results_meteor

def batch_greedy_search(images, questions, model, tokenizer, max_length, device):
    """
    Updated batch greedy search for Qwen2.5-VL
    """
    answers = []

    model.eval()
    with torch.no_grad():
        for img, q in zip(images, questions):
            # Prepare message for single inference (Qwen works better with single samples)
            system_message = (
                "You are a surgical assistant AI for endonasal pituitary surgery. "
                "Rely on visual and textual input to deliver accurate, clinically relevant answers. "
                "Use proper surgical terminology. There are 3 phases, 15 steps, 18 instruments and 14 surgical activities. "
                "Time is measured in minutes. Only short sentence answers."
            )
            
            messages = [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": f"Question: {q}\nAnswer:"}
                    ]
                }
            ]

            # Apply chat template
            text = model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Process with Qwen processor
            inputs = model.processor(
                text=[text],
                images=[img],
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate with the model
            generated_ids = model.model.generate(
                **inputs, 
                max_new_tokens=max_length//2,  # Reasonable limit for answers
                do_sample=False,  # Greedy search
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode only the generated part (exclude the input prompt)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # Decode to text
            output_text = model.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # Clean up the answer
            answer = output_text[0].strip() if output_text else ""
            answers.append(answer)

    return answers


def inference(args, val_loader, model, tokenizer, device):
    references = []
    hypotheses = []

    model.eval()
    with torch.no_grad():
        for i, (img_paths, images, questions, answers) in enumerate(tqdm(val_loader), 0):
            # Use img_paths instead of tensor images
            generated_answers = batch_greedy_search(
                img_paths,  # Changed from images to img_paths
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
    #target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # Qwen-specific modules
    target_modules=["q_proj", "k_proj", "v_proj"]
)

model = PitVQAGen(peft_config=lora_config)
save_dir = f'{args.checkpoint_dir}/{args.best_ckpt_name}'

# Load checkpoint with proper handling of different formats
if os.path.exists(save_dir):
    print(f"Loading model from: {save_dir}")
    checkpoint = torch.load(save_dir, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Best validation loss: {checkpoint.get('best_val_loss', 'unknown'):.6f}")
    else:
        # Legacy format - direct state dict
        model.load_state_dict(checkpoint)
        print("Loaded legacy checkpoint format")
else:
    print(f"ERROR: Model checkpoint not found at {save_dir}")
    print("Please train the model first or check the path.")
    
model.to(device)
model.eval()

# Use the model's own tokenizer instead of GPT2
tokenizer = model.tokenizer

args.batch_size = 8  # Reduced for Qwen2.5-VL due to memory requirements
val_dataset = Pit24VQAClassification(val_seq, folder_head, folder_tail)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers)


references, hypotheses  = inference(args, val_loader=val_dataloader, model=model, tokenizer=tokenizer, device=device)

results_bleu, results_rouge, results_meteor = get_nlp_mettics(references, hypotheses)

print(f"BLEU-1: {results_bleu['precisions'][0]:.6f}, "
      f"BLEU-2: {results_bleu['precisions'][1]:.6f}, "
      f"BLEU-3: {results_bleu['precisions'][2]:.6f}, "
      f"BLEU-4: {results_bleu['precisions'][3]:.6f}, "
      f"Overall BLEU: {results_bleu['bleu']:.6f}")

# Save references and hypotheses to file for detailed analysis
with open(os.path.join(args.checkpoint_dir, "references_Final.txt"), "w") as f_ref, \
    open(os.path.join(args.checkpoint_dir, "hypotheses_Final.txt"), "w") as f_hyp:
    for ref in references:
        f_ref.write(ref + "\n")
    for hyp in hypotheses:
        f_hyp.write(hyp + "\n")
print(f"References and hypotheses saved to {args.checkpoint_dir}")
#print(f"Rouge1: {results_rouge['rouge1']:.6f}")
#print(f"Rouge2: {results_rouge['rouge2']:.6f}")
print(f"RougeL: {results_rouge['rougeL']:.6f}")
#print(f"RougeLsum: {results_rouge['rougeLsum']:.6f}")
print(f"Meteor: {results_meteor['meteor']:.6f}")


print('First 5 Labels:')
print(references[:5])

print('First 5 Prediction:')
print(hypotheses[:5])

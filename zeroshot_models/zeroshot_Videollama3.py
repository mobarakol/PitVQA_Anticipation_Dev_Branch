# import os
# import torch
# import random
# import numpy as np
# from PIL import Image
# import cv2
# import tempfile
# from tqdm import tqdm
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoModelForCausalLM, AutoProcessor

# # --------------- Seed & Env ------------------
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# def seed_everything(seed=42):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(seed)
#     random.seed(seed)
# seed_everything(42)

# # -------- Helper: Save PIL frames to video file -----------
# def save_images_to_video(frames, fps=1):
#     """frames: list of PIL Images; returns temp video file path."""
#     w, h = frames[0].size
#     tmpfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
#     out = cv2.VideoWriter(tmpfile.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#     for img in frames:
#         arr = np.array(img.convert("RGB"))
#         arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
#         out.write(arr)
#     out.release()
#     return tmpfile.name

# # --------------- Dataset Adapter (no change) --------------
# class VideoFrameQADataset(Dataset):
#     def __init__(self, image_root, qa_root, video_sequences, sequence_length=8):
#         self.image_root = image_root
#         self.qa_root = qa_root
#         self.video_sequences = video_sequences
#         self.sequence_length = sequence_length
#         self.samples = self._build_samples()
#         print(f"Built {len(self.samples)} samples from {len(self.video_sequences)} video sequences.")

#     def _build_samples(self):
#         samples = []
#         for seq_id in self.video_sequences:
#             video_folder = f'video_{seq_id.zfill(2)}'
#             image_folder_path = os.path.join(self.image_root, video_folder)
#             if not os.path.isdir(image_folder_path): continue
#             frame_ids = sorted([int(f.split('.')[0]) for f in os.listdir(image_folder_path) if f.endswith('.png')])
#             for i in range(len(frame_ids) - self.sequence_length + 1):
#                 chunk_of_frames = frame_ids[i:i + self.sequence_length]
#                 if chunk_of_frames != list(range(chunk_of_frames[0], chunk_of_frames[0] + self.sequence_length)): continue
#                 frame_paths = [os.path.join(image_folder_path, f"{fid:05d}.png") for fid in chunk_of_frames]
#                 last_frame_id = chunk_of_frames[-1]
#                 qa_file_path = os.path.join(self.qa_root, video_folder, f"{last_frame_id:05d}.txt")
#                 if os.path.isfile(qa_file_path):
#                     samples.append((frame_paths, qa_file_path))
#         return samples

#     def _load_qa_from_file(self, qa_path):
#         qa_pairs = []
#         with open(qa_path, 'r') as f:
#             for line in f:
#                 if "|" in line:
#                     question, answer = line.strip().split("|", 1)
#                     qa_pairs.append((question.strip(), answer.strip()))
#         return qa_pairs

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         frame_paths, qa_path = self.samples[idx]
#         frames = [Image.open(p).convert('RGB') for p in frame_paths]
#         qa_pairs = self._load_qa_from_file(qa_path)
#         return frames, [q for q, a in qa_pairs], [a for q, a in qa_pairs]

# def custom_collate_fn(batch):
#     batch_video_clips, batch_questions, batch_answers = [], [], []
#     for video_clip_frames, questions, answers in batch:
#         for question, answer in zip(questions, answers):
#             batch_video_clips.append(video_clip_frames)
#             batch_questions.append(question)
#             batch_answers.append(answer)
#     return batch_video_clips, batch_questions, batch_answers

# # ----------------- MAIN ------------------
# def main():
#     # ------- Config ---------
#     class Config:
#         batch_size = 1    # VideoLLaMA3 is heavy. 1 is safest
#         num_workers = 0
#         sequence_length = 8
#         image_root = '/SAN/medic/surgicalLLM/content/PitVQA/datasets/PitVQA_Anticipation-25/images'
#         qa_root = '/SAN/medic/surgicalLLM/content/PitVQA/datasets/PitVQA_Anticipation-25/QA_Anticipation'
#         video_sequences = ['01', '02', '03', '04', '05']  # Change as needed

#     cfg = Config()

#     # ------- Load Model/Processor ----------
#     print("Loading VideoLLaMA3-2B...")
#     model_id = "DAMO-NLP-SG/VideoLLaMA3-2B"
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
#     )
#     processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
#     model.eval()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Model loaded on", device)

#     # ------- DataLoader --------------------
#     dataset = VideoFrameQADataset(
#         image_root=cfg.image_root, qa_root=cfg.qa_root,
#         video_sequences=cfg.video_sequences, sequence_length=cfg.sequence_length)
#     dataloader = DataLoader(
#         dataset, batch_size=cfg.batch_size, shuffle=False,
#         num_workers=cfg.num_workers, pin_memory=True, collate_fn=custom_collate_fn)

#     # ------- Inference Loop ----------------
#     all_questions, all_answers, all_predictions = [], [], []

#     for batch_video_clips, batch_questions, batch_answers in tqdm(dataloader):
#         for frames, question, answer in zip(batch_video_clips, batch_questions, batch_answers):
#             # --- Save frames to temporary video file ---
#             video_file = save_images_to_video(frames, fps=1)
#             try:
#                 conversation = [
#                     {"role": "system", "content": "You are a helpful assistant."},
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "video", "video": {"video_path": video_file, "fps": 1, "max_frames": len(frames)}},
#                             {"type": "text", "text": question},
#                         ]
#                     },
#                 ]
#                 # --- Process conversation and generate response ---
#                 inputs = processor(conversation=conversation, return_tensors="pt")
#                 inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
#                 if "pixel_values" in inputs:
#                     inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
#                 with torch.no_grad():
#                     output_ids = model.generate(**inputs, max_new_tokens=64)
#                 response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
#             except Exception as e:
#                 response = f"[ERROR: {str(e)}]"
#             finally:
#                 os.remove(video_file)  # Clean up

#             # --- Store results ---
#             all_questions.append(question)
#             all_answers.append(answer)
#             all_predictions.append(response)
#             print(f"\nQ: {question}\nGT: {answer}\nPred: {response}\n{'-'*40}")

#     # -------- (Optional) Save Results ---------
#     with open("videollama3_zeroshot_results.txt", "w") as f:
#         for q, gt, pred in zip(all_questions, all_answers, all_predictions):
#             f.write(f"Q: {q}\nGT: {gt}\nPred: {pred}\n{'-'*20}\n")

# if __name__ == "__main__":
#     main()


import os
import torch
import random
import numpy as np
from PIL import Image
import cv2
import tempfile
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor

# === NEW: NLP Metrics ===
import evaluate

bleu   = evaluate.load("bleu",   keep_in_memory=True)
rouge  = evaluate.load("rouge",  keep_in_memory=True)
meteor = evaluate.load("meteor", keep_in_memory=True)

def get_nlp_metrics(references, hypotheses):
    results_bleu   = bleu.compute(predictions=hypotheses, references=references)
    results_rouge  = rouge.compute(predictions=hypotheses, references=references)
    results_meteor = meteor.compute(predictions=hypotheses, references=references)
    return results_bleu, results_rouge, results_meteor

def save_lists_as_txt(filename, data):
    with open(filename, 'w') as f:
        for line in data:
            f.write(line.strip() + '\n')

# --------------- Seed & Env ------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
seed_everything(42)

# -------- Helper: Save PIL frames to video file -----------
def save_images_to_video(frames, fps=1):
    w, h = frames[0].size
    tmpfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out = cv2.VideoWriter(tmpfile.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for img in frames:
        arr = np.array(img.convert("RGB"))
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        out.write(arr)
    out.release()
    return tmpfile.name

# --------------- Dataset Adapter --------------
class VideoFrameQADataset(Dataset):
    def __init__(self, image_root, qa_root, video_sequences, sequence_length=8):
        self.image_root = image_root
        self.qa_root = qa_root
        self.video_sequences = video_sequences
        self.sequence_length = sequence_length
        self.samples = self._build_samples()
        print(f"Built {len(self.samples)} samples from {len(self.video_sequences)} video sequences.")

    def _build_samples(self):
        samples = []
        for seq_id in self.video_sequences:
            video_folder = f'video_{seq_id.zfill(2)}'
            image_folder_path = os.path.join(self.image_root, video_folder)
            if not os.path.isdir(image_folder_path): continue
            frame_ids = sorted([int(f.split('.')[0]) for f in os.listdir(image_folder_path) if f.endswith('.png')])
            for i in range(len(frame_ids) - self.sequence_length + 1):
                chunk_of_frames = frame_ids[i:i + self.sequence_length]
                if chunk_of_frames != list(range(chunk_of_frames[0], chunk_of_frames[0] + self.sequence_length)): continue
                frame_paths = [os.path.join(image_folder_path, f"{fid:05d}.png") for fid in chunk_of_frames]
                last_frame_id = chunk_of_frames[-1]
                qa_file_path = os.path.join(self.qa_root, video_folder, f"{last_frame_id:05d}.txt")
                if os.path.isfile(qa_file_path):
                    samples.append((frame_paths, qa_file_path))
        return samples

    def _load_qa_from_file(self, qa_path):
        qa_pairs = []
        with open(qa_path, 'r') as f:
            for line in f:
                if "|" in line:
                    question, answer = line.strip().split("|", 1)
                    qa_pairs.append((question.strip(), answer.strip()))
        return qa_pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, qa_path = self.samples[idx]
        frames = [Image.open(p).convert('RGB') for p in frame_paths]
        qa_pairs = self._load_qa_from_file(qa_path)
        return frames, [q for q, a in qa_pairs], [a for q, a in qa_pairs]

def custom_collate_fn(batch):
    batch_video_clips, batch_questions, batch_answers = [], [], []
    for video_clip_frames, questions, answers in batch:
        for question, answer in zip(questions, answers):
            batch_video_clips.append(video_clip_frames)
            batch_questions.append(question)
            batch_answers.append(answer)
    return batch_video_clips, batch_questions, batch_answers

# ----------------- MAIN ------------------
def main():
    # ------- Config ---------
    class Config:
        batch_size = 1    # VideoLLaMA3 is heavy. 1 is safest
        num_workers = 0
        sequence_length = 8
        image_root = '/SAN/medic/surgicalLLM/content/PitVQA/datasets/PitVQA_Anticipation-25/images'
        qa_root = '/SAN/medic/surgicalLLM/content/PitVQA/datasets/PitVQA_Anticipation-25/QA_Anticipation'
        video_sequences = ['01', '02', '03', '04', '05']  # Change as needed

    cfg = Config()

    # ------- Load Model/Processor ----------
    print("Loading VideoLLaMA3-2B...")
    model_id = "DAMO-NLP-SG/VideoLLaMA3-2B"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Model loaded on", device)

    # ------- DataLoader --------------------
    dataset = VideoFrameQADataset(
        image_root=cfg.image_root, qa_root=cfg.qa_root,
        video_sequences=cfg.video_sequences, sequence_length=cfg.sequence_length)
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=custom_collate_fn)

    # ------- Inference Loop ----------------
    all_questions, all_answers, all_predictions = [], [], []
    system_message = (
        "You are a surgical assistant AI for endonasal pituitary surgery. "
        "Rely on visual and textual input to deliver accurate, clinically relevant answers. "
        "Use proper surgical terminology. There are 3 phases, 15 steps, 18 instruments and 14 surgical activities. "
        "Time is measured in minutes. Only short sentence answers."
    )
    for batch_video_clips, batch_questions, batch_answers in tqdm(dataloader):
        for frames, question, answer in zip(batch_video_clips, batch_questions, batch_answers):
            video_file = save_images_to_video(frames, fps=1)
            try:
                conversation = [
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": {"video_path": video_file, "fps": 1, "max_frames": len(frames)}},
                            {"type": "text", "text": question},
                        ]
                    },
                ]
                inputs = processor(conversation=conversation, return_tensors="pt")
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=64)
                response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            except Exception as e:
                response = f"[ERROR: {str(e)}]"
            finally:
                os.remove(video_file)

            all_questions.append(question)
            all_answers.append(answer)
            all_predictions.append(response)
            print(f"\nQ: {question}\nGT: {answer}\nPred: {response}\n{'-'*40}")

    # --- Save GT and predictions as text ---
    save_lists_as_txt("ground_truth.txt", all_answers)
    save_lists_as_txt("predictions.txt", all_predictions)
    print("Saved ground_truth.txt and predictions.txt.")

    # --- Compute metrics ---
    results_bleu, results_rouge, results_meteor = get_nlp_metrics(all_answers, all_predictions)
    print(f"\n=== NLP Metrics ===")
    print(f"BLEU-1:  {results_bleu['precisions'][0]:.6f}")
    print(f"BLEU-2:  {results_bleu['precisions'][1]:.6f}")
    print(f"BLEU-3:  {results_bleu['precisions'][2]:.6f}")
    print(f"BLEU-4:  {results_bleu['precisions'][3]:.6f}")
    print(f"Overall BLEU: {results_bleu['bleu']:.6f}")
    print(f"ROUGE-1:  {results_rouge['rouge1']:.6f}")
    print(f"ROUGE-2:  {results_rouge['rouge2']:.6f}")
    print(f"ROUGE-L:  {results_rouge['rougeL']:.6f}")
    print(f"ROUGE-Lsum:  {results_rouge['rougeLsum']:.6f}")
    print(f"METEOR:   {results_meteor['meteor']:.6f}")

    # --- Show some random predictions ---
    print("\n=== Sample Predictions ===")
    num_show = min(5, len(all_answers))
    idxs = random.sample(range(len(all_answers)), num_show)
    for i, idx in enumerate(idxs):
        print(f"\nSample {i+1}:")
        print(f"GT:   {all_answers[idx]}")
        print(f"Pred: {all_predictions[idx]}")

if __name__ == "__main__":
    main()

import os
import glob
import time
import gc
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
import evaluate
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login
import numpy as np
import re
from collections import defaultdict
import json
from datetime import datetime

# Set environment variable for HF_HOME
os.environ['HF_HOME'] = '/SAN/medic/surgicalLLM/MedGemmaModels'
hf_home = os.path.expanduser(
    os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
)
print(hf_home)

# Login to Hugging Face
login(token="Your Hugging Face Token Here")  # Replace with your actual token

# Define multiple system messages for testing
SYSTEM_MESSAGES = {
    "original": (
        "You are a surgical assistant AI for endonasal pituitary surgery. Rely on visual and textual input to deliver accurate, clinically relevant answers. Use proper surgical terminology. There are 3 phases, 15 steps, 18 instruments and 14 surgical activities. Time is measured in minutes. Only short sentence answers."
    ),
    
    "v2": (
        "You are a surgical assistant AI for endonasal pituitary surgery. Rely on visual and textual input to deliver accurate, clinically relevant answers. Use proper surgical terminology. Time is measured in minutes. Answers should be a short sentence."
    ),
    
    "v3": (
        "You are a surgical assistant AI for endonasal pituitary surgery. Rely on visual and textual input to deliver accurate, clinically relevant answers. Use proper surgical terminology. Time is measured in minutes. Answers should be a short sentence. Do not reference instruments unless shown visually."
    ), 
}
# Dataset definition (unchanged)
class Pit24VQAClassification(Dataset):
    def __init__(self, seq, folder_head, folder_tail):
        filenames = []
        for curr_seq in seq:
            filenames += glob.glob(folder_head + str(curr_seq) + folder_tail)
        self.vqas = []
        for file in filenames:
            with open(file, "r") as f:
                lines = [line.strip("\n") for line in f if line.strip()]
            for line in lines:
                answer = line.split('|')[1]
                if answer not in ['no_visible_instrument', 'no_secondary_instrument']:
                    self.vqas.append([file, line])
        print('Total files: %d | Total questions: %d' % (len(filenames), len(self.vqas)))

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        qa_full_path = Path(self.vqas[idx][0])
        video_num = qa_full_path.parts[-2]
        file_name = qa_full_path.name

        # Build the image path
        img_loc = os.path.join('/SAN/medic/CARES/mobarak/PitVQA/square_endo_images_Test', video_num, file_name.split('.')[0] + '.png')
        # Load raw image
        raw_image = Image.open(img_loc).convert('RGB')

        # Split question and answer
        question, answer = self.vqas[idx][1].split('|')
        return img_loc, raw_image, question, answer

def format_messages_for_inference(image, question, system_message):
    """Format input for inference with customizable system message"""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image", "image": image}
            ]
        }
    ]

def generate_response(model, processor, image, question, system_message, max_new_tokens=200):
    """Generate response using the customizable system message"""
    messages = format_messages_for_inference(image, question, system_message)
    
    # Apply chat template and tokenize
    inputs = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=True,
        return_dict=True, 
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    
    input_len = inputs["input_ids"].shape[-1]
    
    # Generate response with inference mode
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generation = generation[0][input_len:]
    
    # Decode the response
    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded

def categorize_reference(ref):
    """Categorize reference answers for detailed evaluation"""
    groups = {}
    ref_lower = ref.lower()

    if "next phase" in ref_lower or "remaining phases" in ref_lower:
        groups["newphase"] = ref
    elif ("phase" in ref_lower or "phases" in ref_lower) and not (
        "next phase" in ref_lower or "remaining phases" in ref_lower or "minutes" in ref_lower
    ):
        groups["phase"] = ref
    elif "required" in ref_lower:
        groups["newinstrument"] = ref
    elif "next step" in ref_lower or "remaining steps" in ref_lower:
        groups["newstep"] = ref
    elif ("step" in ref_lower or "steps" in ref_lower) and not (
        "next step" in ref_lower or "remaining steps" in ref_lower or 
        "next steps" in ref_lower or "minutes" in ref_lower or "required" in ref_lower
    ):
        groups["step"] = ref
    elif "activity" in ref_lower:
        groups["activity"] = ref
    elif ("instrument" in ref_lower) and "instruments" not in ref_lower:
        groups["instrument"] = ref
    elif "instruments" in ref_lower:
        groups["quantity"] = ref
    elif "minutes" in ref_lower or "time" in ref_lower:
        groups["time"] = ref
    elif not groups:
        groups["other"] = ref

    return groups

def extract_time(text):
    """Extract time values from text and convert to minutes"""
    total_minutes = 0.0
    pattern = r"([-+]?[0-9]*\.?[0-9]+)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b"
    
    for val, unit in re.findall(pattern, text, re.IGNORECASE):
        v = float(val)
        u = unit.lower()
        if u.startswith("h"):
            total_minutes += v * 60
        elif u.startswith("m"):
            total_minutes += v
        elif u.startswith("s"):
            total_minutes += v / 60

    if total_minutes > 0:
        return total_minutes

    lone_num = re.search(r"([-+]?[0-9]*\.?[0-9]+)\b", text)
    if lone_num:
        return float(lone_num.group(1))

    return None

def get_nlp_metrics(references, hypotheses):
    """Compute NLP evaluation metrics"""
    bleu = evaluate.load("bleu", keep_in_memory=True)
    rouge = evaluate.load("rouge", keep_in_memory=True)
    meteor = evaluate.load("meteor", keep_in_memory=True)
    
    results_bleu = bleu.compute(predictions=hypotheses, references=references)
    results_rouge = rouge.compute(predictions=hypotheses, references=references)
    results_meteor = meteor.compute(predictions=hypotheses, references=references)
    
    return results_bleu, results_rouge, results_meteor

def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

def save_results_to_file(results, filename, output_dir="system_message_results"):
    """Save results to JSON file in specified directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine directory and filename
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {filepath}")

def run_single_system_message_test(model, processor, dataset, system_message_name, system_message_text, max_samples=None):
    """Run inference test for a single system message"""
    print(f"\n{'='*60}")
    print(f"Testing System Message: {system_message_name}")
    print(f"{'='*60}")
    print(f"System Message Content: {system_message_text}")
    print(f"{'='*60}")
    
    all_predictions = []
    all_references = []
    sample_details = []
    
    # Use full dataset if max_samples is None
    total_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    print(f"Processing {total_samples} samples...")
    model.eval()
    
    for idx, (img_loc, image, question, answer) in enumerate(dataset):
        if max_samples and idx >= max_samples:
            break
            
        # Generate response
        start_time = time.time()
        prediction = generate_response(model, processor, image, question, system_message_text)
        inference_time = time.time() - start_time
        
        all_predictions.append(prediction)
        all_references.append(answer)
        
        # Store sample details
        sample_details.append({
            "index": idx,
            "image_path": img_loc,
            "question": question,
            "reference": answer,
            "prediction": prediction,
            "inference_time": inference_time
        })
        
        # Print progress
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{total_samples} samples...")

    # Compute overall metrics
    results_bleu, results_rouge, results_meteor = get_nlp_metrics(all_references, all_predictions)
    
    overall_metrics = {
        "bleu_1": results_bleu['precisions'][0],
        "bleu_2": results_bleu['precisions'][1],
        "bleu_3": results_bleu['precisions'][2],
        "bleu_4": results_bleu['precisions'][3],
        "bleu_overall": results_bleu['bleu'],
        "rouge1": results_rouge['rouge1'],
        "rouge2": results_rouge['rouge2'],
        "rougeL": results_rouge['rougeL'],
        "rougeLsum": results_rouge['rougeLsum'],
        "meteor": results_meteor['meteor']
    }
    
    # Categorical evaluation
    cat_result_dict = defaultdict(lambda: {"references": [], "hypotheses": []})
    
    for ref, hyp in zip(all_references, all_predictions):
        categories = categorize_reference(ref)
        for category in categories:
            cat_result_dict[category]["references"].append(ref)
            cat_result_dict[category]["hypotheses"].append(hyp)

    categorical_metrics = {}
    for category, data in cat_result_dict.items():
        refs_cat = data["references"]
        hyps_cat = data["hypotheses"]
        results_bleu, results_rouge, results_meteor = get_nlp_metrics(refs_cat, hyps_cat)
        
        categorical_metrics[category] = {
            "count": len(refs_cat),
            "bleu": results_bleu['bleu'],
            "rouge1": results_rouge['rouge1'],
            "rougeL": results_rouge['rougeL'],
            "meteor": results_meteor['meteor']
        }

    # Time category MSE
    time_mse = None
    if "time" in cat_result_dict:
        time_refs = cat_result_dict["time"]["references"]
        time_hyps = cat_result_dict["time"]["hypotheses"]
        
        ref_times = [extract_time(ref) for ref in time_refs]
        hyp_times = [extract_time(hyp) for hyp in time_hyps]
        
        valid_pairs = [(r, h) for r, h in zip(ref_times, hyp_times) if r is not None and h is not None]
        
        if valid_pairs:
            squared_errors = [(r - h) ** 2 for r, h in valid_pairs]
            time_mse = np.mean(squared_errors)

    # Calculate average inference time
    avg_inference_time = np.mean([detail["inference_time"] for detail in sample_details])
    
    return {
        "system_message_name": system_message_name,
        "system_message_text": system_message_text,
        "total_samples": total_samples,
        "overall_metrics": overall_metrics,
        "categorical_metrics": categorical_metrics,
        "time_mse": time_mse,
        "avg_inference_time": avg_inference_time,
        "sample_details": sample_details[:10],  # Save first 10 samples for inspection
        "timestamp": datetime.now().isoformat()
    }

def print_results_summary(result):
    """Print a summary of results for one system message"""
    print(f"\n=== Results for {result['system_message_name']} ===")
    metrics = result['overall_metrics']
    print(f"Overall Metrics:")
    print(f"  BLEU-1: {metrics['bleu_1']:.6f}, BLEU-2: {metrics['bleu_2']:.6f}, "
          f"BLEU-3: {metrics['bleu_3']:.6f}, BLEU-4: {metrics['bleu_4']:.6f}")
    print(f"  Overall BLEU: {metrics['bleu_overall']:.6f}")
    print(f"  Rouge1: {metrics['rouge1']:.6f}, Rouge2: {metrics['rouge2']:.6f}, "
          f"RougeL: {metrics['rougeL']:.6f}")
    print(f"  Meteor: {metrics['meteor']:.6f}")
    print(f"  Avg Inference Time: {result['avg_inference_time']:.3f}s")
    
    if result['time_mse'] is not None:
        print(f"  Time MSE: {result['time_mse']:.6f}")
    
    print(f"\nCategorical Results:")
    for category, cat_metrics in result['categorical_metrics'].items():
        print(f"  {category} (n={cat_metrics['count']}): "
              f"BLEU={cat_metrics['bleu']:.4f}, "
              f"Rouge1={cat_metrics['rouge1']:.4f}, "
              f"Meteor={cat_metrics['meteor']:.4f}")

def compare_system_messages(all_results):
    """Compare results across different system messages"""
    print(f"\n{'='*80}")
    print("COMPARISON ACROSS SYSTEM MESSAGES")
    print(f"{'='*80}")
    
    # Create comparison table
    print(f"{'System Message':<20} {'BLEU':<10} {'Rouge1':<10} {'RougeL':<10} {'Meteor':<10} {'Time(s)':<10}")
    print("-" * 80)
    
    for result in all_results:
        metrics = result['overall_metrics']
        print(f"{result['system_message_name']:<20} "
              f"{metrics['bleu_overall']:<10.4f} "
              f"{metrics['rouge1']:<10.4f} "
              f"{metrics['rougeL']:<10.4f} "
              f"{metrics['meteor']:<10.4f} "
              f"{result['avg_inference_time']:<10.3f}")
    
    # Find best performing system message for each metric
    print(f"\nBest Performance by Metric:")
    metrics_to_compare = ['bleu_overall', 'rouge1', 'rougeL', 'meteor']
    
    for metric in metrics_to_compare:
        best_result = max(all_results, key=lambda x: x['overall_metrics'][metric])
        print(f"  {metric}: {best_result['system_message_name']} "
              f"({best_result['overall_metrics'][metric]:.6f})")

# Main execution
def main():
    # Initialize dataset
    val_seq = ['02', '06', '12', '13', '24']
    folder_head = '/SAN/medic/surgicalLLM/content/PitVQA/datasets/PitVQA_Anticipation-25/QA_Test/video_'
    folder_tail = '/*.txt'
    dataset = Pit24VQAClassification(val_seq, folder_head, folder_tail)

    # Load model and processor
    model_id = "google/medgemma-4b-it"
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        force_download=True
    )
    processor = AutoProcessor.from_pretrained(model_id, force_download=True)

    # Configuration
    max_samples = None  # Set to None for full dataset, or specify number for testing
    output_directory = "/SAN/medic/surgicalLLM/content/zeroshot-test/results/medgemma_results"  # Specify output directory
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    print(f"Results will be saved to: {output_directory}")
    
    # Run tests for all system messages
    all_results = []
    
    for msg_name, msg_text in SYSTEM_MESSAGES.items():
        try:
            result = run_single_system_message_test(
                model, processor, dataset, msg_name, msg_text, max_samples
            )
            all_results.append(result)
            print_results_summary(result)
            
            # Save individual result
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_results_to_file(result, f"medgemma_results_{msg_name}_{timestamp}.json", output_directory)
            
            # Clear memory between tests
            clear_memory()
            
        except Exception as e:
            print(f"Error testing {msg_name}: {str(e)}")
            continue
    
    # Save all results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_results_to_file(all_results, f"medgemma_all_system_message_results_{timestamp}.json", output_directory)
    
    # Compare results
    if all_results:
        compare_system_messages(all_results)
    
    print(f"\nTesting completed! Results saved to files.")

if __name__ == "__main__":
    main()
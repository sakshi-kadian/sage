import json
import os
import random
from datasets import load_dataset

def download_shp_subset(num_samples=500, seed=42):
    print("Loading Stanford Human Preferences (SHP) dataset from HuggingFace...")
    # Load the test split of the SHP dataset
    # SHP is quite large, so we just stream it or take a subset
    dataset = load_dataset("stanfordnlp/SHP", split="test")
    
    print(f"Total test samples available: {len(dataset)}")
    
    # Shuffle and select the subset
    dataset = dataset.shuffle(seed=seed)
    subset = dataset.select(range(min(num_samples, len(dataset))))
    
    processed_data = []
    
    for item in subset:
        # SHP structure:
        # 'history': the context/prompt
        # 'human_ref_A': response A
        # 'human_ref_B': response B
        # 'labels': 1 if A is preferred, 0 if B is preferred
        
        prompt = item['history']
        
        if item['labels'] == 1:
            chosen = item['human_ref_A']
            rejected = item['human_ref_B']
        else:
            chosen = item['human_ref_B']
            rejected = item['human_ref_A']
            
        # Format it exactly like our HH-RLHF data so we don't have to rewrite the dataloader!
        processed_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })
        
    output_path = os.path.join(os.path.dirname(__file__), "shp_test_subset.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully saved {len(processed_data)} SHP examples to {output_path}")

if __name__ == "__main__":
    random.seed(42)
    download_shp_subset(num_samples=500)

import os
import json
from datasets import load_dataset

def main():
    print("Downloading Anthropic HH-RLHF dataset...")
    # Load the test split
    dataset = load_dataset("Anthropic/hh-rlhf", split="test")
    
    print(f"Loaded {len(dataset)} examples. Processing 200 for local evaluation...")
    
    processed_data = []
    # Take first 200 examples
    for i in range(200):
        item = dataset[i]
        
        # Extract just the assistant's final response from the chosen/rejected texts
        # The HH-RLHF dataset formats dialogue like "\n\nHuman: ... \n\nAssistant: ..."
        try:
            chosen_response = item['chosen'].split("\n\nAssistant: ")[-1].strip()
            rejected_response = item['rejected'].split("\n\nAssistant: ")[-1].strip()
            
            # Only keep reasonable length responses for attack evaluation
            if 10 < len(chosen_response.split()) < 100:
                processed_data.append({
                    "chosen": chosen_response,
                    "rejected": rejected_response
                })
        except Exception:
            continue
            
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "test_data.json")
    
    with open(out_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
        
    print(f"Saved {len(processed_data)} valid examples to {out_path}")

if __name__ == "__main__":
    main()

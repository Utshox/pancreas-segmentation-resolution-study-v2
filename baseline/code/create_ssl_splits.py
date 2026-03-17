
import os
import glob
import random
import json
import argparse
from pathlib import Path

def create_splits(data_dir, output_dir, seed=42):
    random.seed(seed)
    
    # Get all patient IDs
    x_files = sorted(glob.glob(os.path.join(data_dir, "*_x.npy")))
    patient_ids = [os.path.basename(f).replace("_x.npy", "") for f in x_files]
    
    print(f"Found {len(patient_ids)} patient cases.")
    
    # Shuffle
    random.shuffle(patient_ids)
    
    # Reserve 10 for validation
    val_cases = patient_ids[:10]
    train_cases = patient_ids[10:]
    
    splits = {
        "validation": val_cases,
        "train_pool": train_cases
    }
    
    ratios = [0.1, 0.25, 0.5]
    for r in ratios:
        n_labeled = int(len(train_cases) * r)
        labeled_cases = train_cases[:n_labeled]
        unlabeled_cases = train_cases[n_labeled:]
        
        splits[f"labeled_{int(r*100)}"] = labeled_cases
        splits[f"unlabeled_{int(r*100)}"] = unlabeled_cases
        
        print(f"Ratio {r*100}%: {len(labeled_cases)} labeled, {len(unlabeled_cases)} unlabeled.")

    # Save splits to JSON
    output_path = os.path.join(output_dir, "ssl_splits.json")
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=4)
    
    print(f"Splits saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='baseline/code')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    create_splits(args.data_dir, args.output_dir, args.seed)

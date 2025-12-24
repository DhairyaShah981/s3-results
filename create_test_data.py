#!/usr/bin/env python3
"""
Create test data from FlashRAG datasets
Ensures proper formatting of answers as lists for multi-answer evaluation
"""

import datasets
import pandas as pd
import os
import json

def ensure_list(value):
    """Ensure value is a list of strings"""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]

def main():
    print("=" * 60)
    print("Creating Test Data from FlashRAG Datasets")
    print("=" * 60)
    print()
    
    os.makedirs("data/nq_hotpotqa_train", exist_ok=True)
    
    # Load datasets from FlashRAG
    print("Loading NQ dataset...")
    nq_dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')
    nq_test = nq_dataset['test'] if 'test' in nq_dataset else nq_dataset['dev']
    print(f"  Loaded {len(nq_test)} NQ samples")
    
    # Show sample structure
    if len(nq_test) > 0:
        sample = nq_test[0]
        print(f"  Sample fields: {list(sample.keys())}")
        if 'golden_answers' in sample:
            print(f"  Sample golden_answers: {sample['golden_answers']}")
    
    print("\nLoading HotpotQA dataset...")
    hotpot_dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'hotpotqa')
    hotpot_test = hotpot_dataset['dev']  # HotpotQA only has train/dev, no test
    print(f"  Loaded {len(hotpot_test)} HotpotQA samples")
    
    # Sample subset for faster testing
    print("\nSampling test data...")
    nq_sample = nq_test.select(range(min(50, len(nq_test))))
    hotpot_sample = hotpot_test.select(range(min(50, len(hotpot_test))))
    
    # Process NQ
    print("\nProcessing NQ samples...")
    nq_data = []
    for idx, item in enumerate(nq_sample):
        # Get answers - ensure they're a proper list
        raw_answers = item.get('golden_answers', item.get('answer', []))
        answers = ensure_list(raw_answers)
        
        if not answers:
            print(f"  Warning: No answers for NQ item {idx}: {item['question'][:50]}...")
            answers = ["N/A"]
        
        nq_data.append({
            'data_source': 'nq',
            'prompt': f'<question>{item["question"]}</question>\n\nPlease search for information to answer this question.\n<query>',
            'question': item['question'],
            'answers': answers,  # Always a list now
            'id': f'nq_{idx}'
        })
    
    # Process HotpotQA
    print("Processing HotpotQA samples...")
    hotpot_data = []
    for idx, item in enumerate(hotpot_sample):
        # Get answers - ensure they're a proper list
        raw_answers = item.get('golden_answers', item.get('answer', []))
        answers = ensure_list(raw_answers)
        
        if not answers:
            print(f"  Warning: No answers for HotpotQA item {idx}: {item['question'][:50]}...")
            answers = ["N/A"]
        
        hotpot_data.append({
            'data_source': 'hotpotqa',
            'prompt': f'<question>{item["question"]}</question>\n\nPlease search for information to answer this question.\n<query>',
            'question': item['question'],
            'answers': answers,  # Always a list now
            'id': f'hotpotqa_{idx}'
        })
    
    # Combine and save
    all_data = nq_data + hotpot_data
    df = pd.DataFrame(all_data)
    
    print(f"\n✓ Created {len(df)} test samples ({len(nq_data)} NQ + {len(hotpot_data)} HotpotQA)")
    
    # Show sample data to verify correctness
    print("\n📋 Sample Data Verification:")
    for i, row in df.head(5).iterrows():
        print(f"  [{row['id']}] Q: {row['question'][:60]}...")
        print(f"       A: {row['answers']}")
    
    # Save full version
    output_file = 'data/nq_hotpotqa_train/test_e5_s3.parquet'
    df.to_parquet(output_file)
    print(f"\n✓ Saved to {output_file}")
    
    # Also save a smaller sampled version for quick testing
    df_sampled = df.head(20)
    sampled_file = 'data/nq_hotpotqa_train/test_e5_s3_sampled.parquet'
    df_sampled.to_parquet(sampled_file)
    print(f"✓ Saved sampled version (20 samples) to {sampled_file}")
    
    # Save human-readable JSON for verification
    json_file = 'data/nq_hotpotqa_train/test_data_preview.json'
    preview_data = [
        {
            'id': row['id'],
            'question': row['question'],
            'answers': row['answers'],
            'data_source': row['data_source']
        }
        for _, row in df.head(10).iterrows()
    ]
    with open(json_file, 'w') as f:
        json.dump(preview_data, f, indent=2)
    print(f"✓ Saved preview to {json_file}")
    
    print("\n" + "=" * 60)
    print("✓ Test data creation complete!")
    print("=" * 60)
    print("\nYou can now run: bash run_full_inference.sh")

if __name__ == "__main__":
    main()


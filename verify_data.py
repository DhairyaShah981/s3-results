#!/usr/bin/env python3
"""
Verify test data format and show sample questions with all their ground truth answers
"""

import pandas as pd
import os
import json

def main():
    print("=" * 70)
    print("📋 Verifying Test Data Format")
    print("=" * 70)
    
    # Check for data files
    data_files = [
        'data/nq_hotpotqa_train/test_e5_s3.parquet',
        'data/nq_hotpotqa_train/test_e5_s3_sampled.parquet'
    ]
    
    for data_file in data_files:
        if not os.path.exists(data_file):
            print(f"\n⚠️  File not found: {data_file}")
            continue
            
        print(f"\n📁 {data_file}")
        print("-" * 70)
        
        df = pd.read_parquet(data_file)
        print(f"   Total samples: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
        # Check answer format
        print(f"\n   Answer Format Check:")
        for idx, row in df.head(10).iterrows():
            question = row['question']
            answers = row['answers']
            
            # Check if answers is properly formatted
            if hasattr(answers, 'tolist'):
                answers = answers.tolist()
            
            is_list = isinstance(answers, list)
            num_answers = len(answers) if is_list else 1
            
            status = "✓" if is_list and num_answers > 0 else "⚠️"
            
            print(f"\n   {status} [{row['id']}]")
            print(f"      Q: {question[:70]}...")
            print(f"      A: {answers} (type: {type(answers).__name__}, count: {num_answers})")
    
    # Show detailed examples from results if available
    results_file = 'data/output_full_inference/final_results.json'
    if os.path.exists(results_file):
        print(f"\n{'='*70}")
        print("📊 Previous Results Analysis")
        print("=" * 70)
        
        with open(results_file) as f:
            data = json.load(f)
        
        if 'summary' in data:
            summary = data['summary']
            print(f"\n   Summary:")
            print(f"      Total: {summary.get('total_samples', 'N/A')}")
            print(f"      Correct: {summary.get('correct', 'N/A')}")
            print(f"      Accuracy: {summary.get('accuracy', 'N/A'):.1f}%")
        
        if 'results' in data:
            print(f"\n   Sample Results:")
            for result in data['results'][:5]:
                print(f"\n   [{result['id']}]")
                print(f"      Q: {result['question'][:60]}...")
                print(f"      Generated: {result['answer'][:60]}...")
                print(f"      GT Options: {result.get('ground_truth_all', [result['ground_truth']])}")
                print(f"      Correct: {'✓' if result.get('is_correct') else '✗'}")
    
    print("\n" + "=" * 70)
    print("✓ Verification complete!")
    print("=" * 70)
    
    print("""
Next Steps:
1. If answers aren't lists, regenerate test data:
   python3 create_test_data.py

2. Run inference with fixed evaluation:
   python3 full_inference_single_gpu.py

3. Results will show:
   - ALL possible ground truth answers
   - Which answer was matched (if any)
   - Detailed matching scores
""")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Simple inference script for s3 model - bypasses the RL training framework
Uses only the Actor model for generation (no Critic, no Reference, no vLLM)
Memory usage: ~15-20GB instead of ~95-100GB
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
import os
from tqdm import tqdm

def load_model_and_tokenizer(model_path, device="cuda"):
    """Load the trained s3 model"""
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Model loaded: {model.num_parameters() / 1e9:.2f}B parameters")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    return model, tokenizer

def generate_search_query(model, tokenizer, question, max_new_tokens=100):
    """Generate a search query for the given question"""
    
    # Format prompt for search query generation
    prompt = f"<question>{question}</question>\n\nPlease search for information to answer this question.\n<query>"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract query between <query> and </query> or end
    if "<query>" in generated_text:
        query_part = generated_text.split("<query>")[-1]
        if "</query>" in query_part:
            query = query_part.split("</query>")[0].strip()
        else:
            query = query_part.strip()
    else:
        query = generated_text.strip()
    
    return query

def generate_answer(model, tokenizer, question, search_query, max_new_tokens=200):
    """Generate an answer based on the question and search query"""
    
    # Format prompt for answer generation
    prompt = f"<question>{question}</question>\n\n<search_query>{search_query}</search_query>\n\nBased on the search query, please provide an answer:\n<answer>"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract answer
    if "<answer>" in generated_text:
        answer_part = generated_text.split("<answer>")[-1]
        if "</answer>" in answer_part:
            answer = answer_part.split("</answer>")[0].strip()
        else:
            answer = answer_part.strip()
    else:
        answer = generated_text.strip()
    
    return answer

def main():
    print("=" * 60)
    print("s3 Simple Inference - Direct Model Generation")
    print("=" * 60)
    print()
    
    # Configuration
    model_path = "verl_checkpoints/s3_8_3_3_42/actor/global_step_20"
    data_path = "data/nq_hotpotqa_train/test_e5_s3.parquet"
    output_dir = "data/output_simple_inference"
    num_samples = 5
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Load test data
    print(f"\nLoading test data from {data_path}...")
    df = pd.read_parquet(data_path)
    df = df.head(num_samples)
    print(f"Loaded {len(df)} samples")
    
    # Run inference
    print(f"\nRunning inference on {num_samples} samples...")
    print()
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        question = row['question']
        # Handle answers - convert numpy array to list if needed
        answers = row['answers']
        if hasattr(answers, 'tolist'):  # numpy array
            answers = answers.tolist()
        ground_truth = answers[0] if isinstance(answers, list) else str(answers)
        
        print(f"\n{'='*60}")
        print(f"Sample {idx + 1}/{len(df)}")
        print(f"{'='*60}")
        print(f"Question: {question}")
        
        # Generate search query
        search_query = generate_search_query(model, tokenizer, question)
        print(f"Search Query: {search_query}")
        
        # Generate answer
        answer = generate_answer(model, tokenizer, question, search_query)
        print(f"Generated Answer: {answer}")
        print(f"Ground Truth: {ground_truth}")
        
        # Save result
        result = {
            'id': str(row.get('id', f'sample_{idx}')),
            'question': str(question),
            'search_query': str(search_query),
            'generated_answer': str(answer),
            'ground_truth': str(ground_truth),
            'data_source': str(row.get('data_source', 'unknown'))
        }
        results.append(result)
        
        # Save individual result
        with open(f"{output_dir}/sample_{idx}.json", 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Save all results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/all_results.csv", index=False)
    results_df.to_json(f"{output_dir}/all_results.json", orient='records', indent=2)
    
    print(f"\n{'='*60}")
    print("✓ Inference Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}/")
    print(f"- Individual results: sample_*.json")
    print(f"- All results: all_results.csv, all_results.json")
    print()
    print(f"Final memory usage: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print()

if __name__ == "__main__":
    main()


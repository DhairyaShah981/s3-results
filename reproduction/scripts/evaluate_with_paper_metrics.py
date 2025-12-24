#!/usr/bin/env python3
"""
S3 Evaluation Script using Paper's Exact Metrics
Author: Dhairya Shah

Metrics (matching s3/verl/utils/reward_score/rag_2.py):
- Accuracy: answer_span_check + check_if_response_is_correct_llm (semantic)
- Exact Match (EM): Normalized string equality

Usage:
    python3 evaluate_with_paper_metrics.py --input_dir data/output_quick_test
"""

import os
import sys
import json
import re
import string
import argparse
from tqdm import tqdm
import requests
import time

# Try to import pyserini tokenizer, fallback to simple tokenizer
try:
    from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
    _tokenizer = SimpleTokenizer()
    HAS_PYSERINI = True
except ImportError:
    print("[Warning] pyserini not available, using simple tokenizer")
    HAS_PYSERINI = False
    _tokenizer = None


def flatten_answers(answers):
    """Flatten nested lists and convert all to strings"""
    if answers is None:
        return []
    if isinstance(answers, str):
        return [answers]
    
    result = []
    for item in answers:
        if isinstance(item, list):
            result.extend(flatten_answers(item))
        elif item is not None:
            result.append(str(item))
    return result


def normalize_answer(s):
    """Normalize answer string - from paper's rag_2.py"""
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    """Exact Match check - from paper's rag_2.py"""
    golden_answers = flatten_answers(golden_answers)
    if not golden_answers:
        return 0
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return 1
    return 0


def answer_span_check(prediction, golden_answers):
    """Span-based answer check - from paper's rag_2.py"""
    golden_answers = flatten_answers(golden_answers)
    if not golden_answers:
        return 0
    normalized_prediction = normalize_answer(prediction)
    normalized_golden_answers = [normalize_answer(ga) for ga in golden_answers]
    
    if HAS_PYSERINI:
        if has_answers(normalized_prediction, normalized_golden_answers, _tokenizer, regex=False):
            return 1
    else:
        for ga in normalized_golden_answers:
            if ga in normalized_prediction or normalized_prediction in ga:
                return 1
    return 0


def call_llm(prompt, api_url, model, max_tokens=10, temperature=0.0, retries=3):
    """Call LLM API"""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 1.0,
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(
                f"{api_url.rstrip('/')}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            pass
        time.sleep(2 ** attempt)
    return ""


def check_if_response_is_correct_llm(response, gold_answers, api_url, model):
    """LLM-based semantic check - from paper's local_inst.py"""
    prompt = f"Please check if any of the golden answers is contained in the following response: {response}\n\nGolden answers: {str(gold_answers)}\n\nPlease directly answer with 'yes' or 'no'."
    yes_or_no = call_llm(prompt, api_url, model)
    return "yes" in yes_or_no.lower()


def check_answer_correct(answer, golden_answers, api_url, model):
    """Combined answer check - from paper's rag_2.py"""
    golden_answers = flatten_answers(golden_answers)
    if not golden_answers:
        return 0
    
    # First: span check (fast)
    if answer_span_check(answer, golden_answers):
        return 1
    
    # Second: LLM semantic check
    if check_if_response_is_correct_llm(answer, golden_answers, api_url, model):
        return 1
    
    return 0


def generate_answer(question, context, api_url, model, max_tokens=300):
    """Generate answer using the Generator LLM"""
    system_message = f"""Use the following contexts (some might be irrelevant) on demand:
Contexts:
{context}

Important: You MUST directly answer the question without any other text and thinking."""
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    
    try:
        response = requests.post(
            f"{api_url.rstrip('/')}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        return f"[ERROR: {response.status_code}]"
    except Exception as e:
        return f"[ERROR: {e}]"


def run_evaluation(input_dir, api_url, model):
    """Run evaluation using paper's exact metrics"""
    
    print("=" * 60)
    print("S3 EVALUATION (Paper Metrics)")
    print("=" * 60)
    print(f"Input:     {input_dir}")
    print(f"API:       {api_url}")
    print(f"Model:     {model}")
    print(f"Metrics:   EM (exact match) + Accuracy (span + LLM semantic)")
    print("=" * 60)
    
    json_files = [f for f in os.listdir(input_dir) if f.endswith('_output_sequences.json')]
    if not json_files:
        print("No output sequence JSON files found!")
        return
    
    print(f"\nFound {len(json_files)} dataset files\n")
    
    stats = {}
    total_correct = 0
    total_em = 0
    total_questions = 0
    all_results = []
    
    for json_file in json_files:
        data_source = json_file.replace('_output_sequences.json', '')
        
        with open(os.path.join(input_dir, json_file), 'r') as f:
            data = json.load(f)
        
        correct = 0
        em_correct = 0
        source_results = []
        
        print(f"\n{data_source}:")
        print("-" * 40)
        
        for question, entry in tqdm(data.items(), desc=data_source):
            golden_answers = entry.get("golden_answers", [])
            context = entry.get("context_with_info", "")
            
            if len(context) > 8000:
                context = context[:8000] + "..."
            
            model_output = generate_answer(question, context, api_url, model)
            
            if model_output.startswith("[ERROR"):
                source_results.append({
                    "question": question,
                    "golden_answers": golden_answers,
                    "model_output": model_output,
                    "em": 0,
                    "accuracy": 0,
                    "data_source": data_source
                })
                total_questions += 1
                continue
            
            is_correct = check_answer_correct(model_output, golden_answers, api_url, model)
            is_em = em_check(model_output, golden_answers)
            
            if is_correct:
                correct += 1
                total_correct += 1
            if is_em:
                em_correct += 1
                total_em += 1
            total_questions += 1
            
            source_results.append({
                "question": question,
                "golden_answers": golden_answers,
                "model_output": model_output,
                "em": is_em,
                "accuracy": is_correct,
                "data_source": data_source
            })
        
        num_questions = len(data)
        accuracy = correct / num_questions if num_questions > 0 else 0
        em_accuracy = em_correct / num_questions if num_questions > 0 else 0
        
        stats[data_source] = {
            'total': num_questions,
            'correct': correct,
            'accuracy': accuracy,
            'em_correct': em_correct,
            'em_accuracy': em_accuracy
        }
        
        print(f"  Total:    {num_questions}")
        print(f"  Accuracy: {correct}/{num_questions} ({accuracy:.2%})")
        print(f"  EM:       {em_correct}/{num_questions} ({em_accuracy:.2%})")
        
        all_results.extend(source_results)
    
    # Overall stats
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
    overall_em = total_em / total_questions if total_questions > 0 else 0
    
    stats['overall'] = {
        'total': total_questions,
        'correct': total_correct,
        'accuracy': overall_accuracy,
        'em_correct': total_em,
        'em_accuracy': overall_em
    }
    
    print("\n" + "=" * 60)
    print("OVERALL RESULTS (Paper Metrics)")
    print("=" * 60)
    print(f"Total Questions: {total_questions}")
    print(f"Accuracy:        {total_correct}/{total_questions} ({overall_accuracy:.2%})")
    print(f"  (span check + LLM semantic check)")
    print(f"Exact Match:     {total_em}/{total_questions} ({overall_em:.2%})")
    print(f"  (normalized string equality)")
    print("=" * 60)
    
    # Save results
    results_file = os.path.join(input_dir, "evaluation_results_paper_metrics.json")
    with open(results_file, 'w') as f:
        json.dump({"stats": stats, "detailed_results": all_results}, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Show samples
    print("\n" + "=" * 60)
    print("SAMPLE RESULTS (first 3)")
    print("=" * 60)
    for i, r in enumerate(all_results[:3], 1):
        print(f"\n[{i}] Q: {r['question'][:60]}...")
        print(f"    Gold: {r['golden_answers']}")
        print(f"    Pred: {r['model_output'][:80]}...")
        print(f"    EM: {'✓' if r['em'] else '✗'} | Accuracy: {'✓' if r['accuracy'] else '✗'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate s3 results using paper's exact metrics")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000")
    
    args = parser.parse_args()
    run_evaluation(args.input_dir, args.api_url, args.model)


if __name__ == "__main__":
    main()


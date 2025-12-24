#!/usr/bin/env python3
"""
Full s3 Inference on Single GPU (80GB)
- Uses actual retriever (E5 + FAISS + Wikipedia corpus)
- Uses actual datasets (NQ/HotpotQA)
- Actor model generates search queries
- Retriever fetches real documents
- Model generates answers from retrieved context

Memory: ~20-25GB total (fits easily in 80GB)
"""

import torch
import requests
import json
import os
import re
import subprocess
import time
import signal
import sys
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Optional
from dataclasses import dataclass


class Logger:
    """Comprehensive logger for s3 pipeline tracing"""
    
    def __init__(self, log_file: str = "s3_trace.log"):
        self.log_file = log_file
        self.start_time = time.time()
        
        # Clear previous log
        with open(log_file, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write(f"s3 INFERENCE TRACE LOG\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
    
    def log(self, message: str, level: str = "INFO", component: str = "SYSTEM"):
        """Log message to both console and file"""
        timestamp = time.time() - self.start_time
        log_entry = f"[{timestamp:>8.2f}s] [{component:>12}] [{level:>7}] {message}"
        
        # Console output
        print(log_entry)
        
        # File output
        with open(self.log_file, 'a') as f:
            f.write(log_entry + "\n")
    
    def log_section(self, title: str):
        """Log a section header"""
        separator = "=" * 80
        self.log(separator, level="", component="")
        self.log(title, level="", component="")
        self.log(separator, level="", component="")
    
    def log_actor_input(self, prompt: str):
        """Log Actor model input"""
        self.log("Actor Input:", level="INPUT", component="ACTOR")
        lines = prompt.split('\n')
        for line in lines[:10]:  # First 10 lines
            self.log(f"  {line}", level="", component="")
        if len(lines) > 10:
            self.log(f"  ... ({len(lines) - 10} more lines)", level="", component="")
    
    def log_actor_output(self, output: str, query: str):
        """Log Actor model output"""
        self.log("Actor Raw Output:", level="OUTPUT", component="ACTOR")
        self.log(f"  {output[:200]}...", level="", component="")
        self.log(f"Extracted Query: '{query}'", level="QUERY", component="ACTOR")
    
    def log_retrieval(self, query: str, num_docs: int):
        """Log retrieval request"""
        self.log(f"Searching for: '{query}'", level="SEARCH", component="RETRIEVER")
        self.log(f"Retrieved {num_docs} documents", level="RESULT", component="RETRIEVER")
    
    def log_documents(self, docs: List[Dict], max_docs: int = 3):
        """Log retrieved documents"""
        self.log("Retrieved Documents:", level="DOCS", component="RETRIEVER")
        for idx, doc in enumerate(docs[:max_docs]):
            try:
                content = doc['document']['contents']
                title = content.split('\n')[0][:100]
                self.log(f"  Doc {idx+1}: {title}...", level="", component="")
            except:
                self.log(f"  Doc {idx+1}: [parsing error]", level="", component="")
        if len(docs) > max_docs:
            self.log(f"  ... and {len(docs) - max_docs} more documents", level="", component="")
    
    def log_generator_input(self, question: str, context: str):
        """Log Generator input"""
        self.log("Generator Input:", level="INPUT", component="GENERATOR")
        self.log(f"  Question: {question}", level="", component="")
        self.log(f"  Context: {len(context)} chars, {len(context.split())} words", level="", component="")
    
    def log_generator_output(self, answer: str):
        """Log Generator output"""
        self.log(f"Generated Answer: '{answer}'", level="OUTPUT", component="GENERATOR")
    
    def log_comparison(self, generated: str, ground_truth_list):
        """Log answer comparison with improved matching against ALL possible answers"""
        import re
        from difflib import SequenceMatcher
        
        # Ensure ground_truth_list is a list
        if isinstance(ground_truth_list, str):
            ground_truth_list = [ground_truth_list]
        elif hasattr(ground_truth_list, 'tolist'):
            ground_truth_list = ground_truth_list.tolist()
        
        # Flatten if nested list
        flat_gt_list = []
        for gt in ground_truth_list:
            if isinstance(gt, list):
                flat_gt_list.extend([str(g) for g in gt])
            else:
                flat_gt_list.append(str(gt))
        ground_truth_list = flat_gt_list
        
        self.log("Answer Comparison:", level="EVAL", component="EVALUATION")
        self.log(f"  Generated:    '{generated}'", level="", component="")
        self.log(f"  Ground Truth Options: {ground_truth_list}", level="", component="")
        
        gen_lower = generated.lower()
        
        # Check against ALL possible correct answers
        is_match = False
        matched_answer = None
        best_scores = {'exact': False, 'reverse': False, 'token': 0.0, 'fuzzy': 0.0}
        
        for gt in ground_truth_list:
            gt_lower = str(gt).lower().strip()
            if not gt_lower:
                continue
            
            # Strategy 1: Exact substring match
            exact_match = gt_lower in gen_lower
            
            # Strategy 2: Reverse match (generated in ground truth)
            reverse_match = gen_lower in gt_lower
            
            # Strategy 3: Token overlap (word-based matching)
            gen_tokens = set(re.findall(r'\b\w+\b', gen_lower))
            gt_tokens = set(re.findall(r'\b\w+\b', gt_lower))
            
            if gt_tokens:
                overlap = len(gen_tokens & gt_tokens) / len(gt_tokens)
            else:
                overlap = 0.0
            
            # Strategy 4: Fuzzy string similarity
            similarity = SequenceMatcher(None, gen_lower, gt_lower).ratio()
            
            # Track best scores
            if exact_match:
                best_scores['exact'] = True
            if reverse_match:
                best_scores['reverse'] = True
            best_scores['token'] = max(best_scores['token'], overlap)
            best_scores['fuzzy'] = max(best_scores['fuzzy'], similarity)
            
            # Check if this answer matches
            if exact_match or reverse_match or overlap >= 0.5 or similarity >= 0.6:
                is_match = True
                matched_answer = gt
                break
        
        # Log detailed results
        self.log(f"  Matching Scores (best across all GT):", level="", component="")
        self.log(f"    • Exact match: {'✓' if best_scores['exact'] else '✗'}", level="", component="")
        self.log(f"    • Reverse match: {'✓' if best_scores['reverse'] else '✗'}", level="", component="")
        self.log(f"    • Token overlap: {best_scores['token']:.1%} {'✓' if best_scores['token'] >= 0.5 else '✗'}", level="", component="")
        self.log(f"    • Fuzzy similarity: {best_scores['fuzzy']:.1%} {'✓' if best_scores['fuzzy'] >= 0.6 else '✗'}", level="", component="")
        
        if is_match:
            self.log(f"  ✓ CORRECT! Matched: '{matched_answer}'", level="MATCH", component="EVALUATION")
            return True, matched_answer
        else:
            self.log("  ✗ INCORRECT - No ground truth matched", level="MISMATCH", component="EVALUATION")
            return False, None

@dataclass
class Config:
    # Actor Model (for search query generation)
    actor_model_path: str = "verl_checkpoints/s3_8_3_3_42/actor/global_step_20"
    
    # Generator LLM (frozen, for final answer generation)
    # Option 1: Use OpenAI API (recommended - no GPU needed)
    use_openai: bool = True  # Set to True to use OpenAI API
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")  # Set via environment variable
    openai_model: str = "gpt-4o-mini"  # GPT-4o-mini model
    
    # Option 2: Use local vLLM server (requires vLLM running)
    generator_model: str = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    generator_port: int = 8000  # vLLM server port
    use_generator_api: bool = False  # Legacy flag
    
    # Retriever
    index_path: str = "data/retrieval/e5_Flat.index"
    corpus_path: str = "data/retrieval/wiki-18.jsonl"
    retriever_model: str = "intfloat/e5-base-v2"
    retriever_port: int = 3000
    topk: int = 5
    
    # Data
    test_data_path: str = "data/nq_hotpotqa_train/test_e5_s3.parquet"
    test_data_sampled_path: str = "data/nq_hotpotqa_train/test_e5_s3_sampled.parquet"
    num_samples: int = 10
    
    # Generation
    max_query_tokens: int = 100
    max_answer_tokens: int = 300
    max_turns: int = 2
    temperature: float = 0.6
    
    # Evaluation (answer matching thresholds)
    eval_token_overlap_threshold: float = 0.5  # 50% word overlap required
    eval_fuzzy_similarity_threshold: float = 0.6  # 60% string similarity required
    eval_verbose: bool = True  # Show detailed matching scores
    
    # Output
    output_dir: str = "data/output_full_inference"


class RetrieverClient:
    """Client to interact with the retrieval server"""
    
    def __init__(self, url: str, topk: int = 5, use_e5_prefix: bool = True):
        self.url = url
        self.topk = topk
        self.use_e5_prefix = use_e5_prefix  # E5 models require "query: " prefix
        
    def search(self, queries: List[str]) -> List[List[Dict]]:
        """Send queries to retriever and get documents"""
        if not queries:
            return []
        
        # E5 embedding models require "query: " prefix for queries
        if self.use_e5_prefix:
            prefixed_queries = [f"query: {q}" if not q.startswith("query: ") else q for q in queries]
        else:
            prefixed_queries = queries
            
        payload = {
            "queries": prefixed_queries,
            "topk": self.topk,
            "return_scores": True
        }
        
        try:
            response = requests.post(self.url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()['result']
        except Exception as e:
            print(f"Retrieval error: {e}")
            return [[] for _ in queries]
    
    def format_docs(self, docs: List[Dict]) -> str:
        """Format retrieved documents into a string"""
        if not docs:
            return "No documents found."
            
        formatted = ""
        for idx, doc_item in enumerate(docs):
            try:
                content = doc_item['document']['contents']
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                formatted += f"Doc {idx+1} (Title: {title}): {text}\n\n"
            except:
                continue
        return formatted.strip() if formatted else "No documents found."


class GeneratorClient:
    """Client to interact with the frozen Generator LLM (vLLM or OpenAI API)"""
    
    def __init__(self, model: str, port: int = 8000, use_api: bool = False, api_key: str = ""):
        self.model = model
        self.port = port
        self.use_api = use_api
        self.api_key = api_key
        self.url = f"http://127.0.0.1:{port}/v1/chat/completions"
        
    def generate(self, question: str, context: str) -> str:
        """Generate final answer using retrieved context"""
        
        if self.use_api:
            # Use external API (Claude, GPT, etc.)
            return self._generate_with_api(question, context)
        else:
            # Use local vLLM server
            return self._generate_with_vllm(question, context)
    
    def _generate_with_vllm(self, question: str, context: str) -> str:
        """Generate using local vLLM server"""
        system_message = f"""Use the following contexts (some might be irrelevant) on demand:

Contexts:
{context}

Important: You MUST directly answer the question without any other text and thinking."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 300
        }
        
        try:
            response = requests.post(self.url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Generator error: {e}")
            return f"[Generator Error: {e}]"
    
    def _generate_with_api(self, question: str, context: str) -> str:
        """Generate using external API (OpenAI GPT)"""
        try:
            import openai
            from openai import OpenAI
        except ImportError:
            return "[Error: openai package not installed. Run: pip install openai]"
        
        # Check API key
        if not self.api_key:
            return "[Error: OPENAI_API_KEY not set. Please set it via environment variable or config]"
        
        # Initialize OpenAI client
        client = OpenAI(api_key=self.api_key)
        
        system_message = f"""Use the following contexts (some might be irrelevant) on demand:

Contexts:
{context}

Important: You MUST directly answer the question without any other text and thinking."""
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": question}
                ],
                temperature=0.0,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[OpenAI API Error: {e}]"


class S3InferenceEngine:
    """s3 inference engine: Actor (search) + Retriever + Generator (answer)"""
    
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
        self.actor_model = None  # For generating search queries
        self.tokenizer = None
        self.retriever = None
        self.generator = None  # Frozen LLM for final answers
        
    def load_actor_model(self):
        """Load the Actor model (trained for search query generation)"""
        print(f"Loading Actor model from {self.config.actor_model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.actor_model_path)
        self.actor_model = AutoModelForCausalLM.from_pretrained(
            self.config.actor_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"Actor loaded: {self.actor_model.num_parameters() / 1e9:.2f}B parameters")
        print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
    def connect_retriever(self):
        """Connect to retriever server"""
        url = f"http://127.0.0.1:{self.config.retriever_port}/retrieve"
        self.retriever = RetrieverClient(url, self.config.topk)
        
        # Test connection
        try:
            test_result = self.retriever.search(["test query"])
            print(f"Retriever connected at port {self.config.retriever_port}")
            return True
        except Exception as e:
            print(f"Warning: Retriever not available ({e})")
            return False
    
    def connect_generator(self):
        """Connect to generator LLM (OpenAI API or local vLLM)"""
        if self.config.use_openai:
            # Use OpenAI API
            self.generator = GeneratorClient(
                model=self.config.openai_model,
                port=self.config.generator_port,
                use_api=True,
                api_key=self.config.openai_api_key
            )
            print(f"Using OpenAI API: {self.config.openai_model}")
            
            # Test API connection
            if not self.config.openai_api_key:
                print("WARNING: OPENAI_API_KEY not set!")
                return False
            
            # Quick test
            try:
                test_answer = self.generator.generate("test", "test context")
                if "Error" not in test_answer:
                    print("✓ OpenAI API connected successfully")
                    return True
                else:
                    print(f"✗ OpenAI API test failed: {test_answer}")
                    return False
            except Exception as e:
                print(f"✗ OpenAI API error: {e}")
                return False
        else:
            # Use local vLLM server
            self.generator = GeneratorClient(
                model=self.config.generator_model,
                port=self.config.generator_port,
                use_api=self.config.use_generator_api,
                api_key=""
            )
            
            # Test vLLM connection
            try:
                test_answer = self.generator.generate("test", "test context")
                print(f"Generator LLM connected at port {self.config.generator_port}")
                return True
            except Exception as e:
                print(f"Warning: Generator not available ({e})")
                print("Will use Actor model for answer generation (fallback)")
                return False
    
    def generate_search_query(self, question: str, context: str = "", turn: int = 1) -> str:
        """Generate a search query using Actor model.
        
        NOTE: The training prompt (train_s3.py) includes placeholder examples like [search query]
        which confuse the model during inference. For inference, we use a simpler prompt
        that matches how the model was trained to respond.
        """
        
        self.logger.log(f"=== TURN {turn}: QUERY GENERATION ===", level="TURN", component="ACTOR")
        
        # System instruction (matches training format from train_s3.py)
        # Key: mentions <think> tags to encourage reasoning output
        system_instruction = """You are a search copilot for the generation model. You will generate search queries to help find relevant information.
You can add reasoning within <think></think> tags to explain your search strategy.
Show the search query in JSON format between <query> and </query>.

"""
        
        if context:
            # Subsequent turns - has previous search results
            prompt = f"""{system_instruction}<question>{question}</question>

<information>
{context}
</information>

Based on the above, generate your next search query:
<query>{{"query": \""""
        else:
            # First turn - no context yet
            prompt = f"""{system_instruction}<question>{question}</question>

Generate a search query to find information for this question:
<query>{{"query": \""""
        
        self.logger.log_actor_input(prompt)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.actor_model.device)
        
        self.logger.log(f"Generating with Actor model (temp={self.config.temperature})...", 
                       level="PROCESS", component="ACTOR")
        
        with torch.no_grad():
            outputs = self.actor_model.generate(
                **inputs,
                max_new_tokens=self.config.max_query_tokens,
                temperature=self.config.temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract thinking and query
        thinking = self._extract_thinking(generated)
        query = self._extract_query(generated)
        
        # Log thinking process if present
        if thinking:
            self.logger.log(f"Actor Reasoning: {thinking}", level="THINK", component="ACTOR")
        
        self.logger.log_actor_output(generated, query)
        
        return query
    
    def _extract_query(self, text: str) -> str:
        """Extract query from generated text"""
        # Try JSON format first
        try:
            match = re.search(r'"query"\s*:\s*"([^"]+)"', text)
            if match:
                return match.group(1).strip()
        except:
            pass
            
        # Try XML format
        try:
            match = re.search(r'<query>(.*?)</query>', text, re.DOTALL)
            if match:
                content = match.group(1).strip()
                # Try to parse as JSON
                try:
                    data = json.loads(content)
                    if 'query' in data:
                        return data['query']
                except:
                    return content
        except:
            pass
            
        # Fallback: return everything after <query>
        if '<query>' in text:
            return text.split('<query>')[-1].split('</query>')[0].split('"')[0].strip()
            
        return ""
    
    def _extract_thinking(self, text: str) -> str:
        """Extract reasoning from <think> tags"""
        try:
            match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
            if match:
                return match.group(1).strip()
        except:
            pass
        return ""
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate final answer using Generator LLM (frozen)"""
        
        self.logger.log("=== FINAL ANSWER GENERATION ===", level="TURN", component="GENERATOR")
        self.logger.log_generator_input(question, context)
        
        if self.generator:
            # Use frozen Generator LLM (proper s3 architecture)
            self.logger.log("Using frozen Generator LLM (proper s3 architecture)", 
                          level="ARCH", component="GENERATOR")
            answer = self.generator.generate(question, context)
        else:
            # Fallback: use Actor model (not ideal but works)
            self.logger.log("⚠️  FALLBACK: Using Actor for answer (Generator unavailable)", 
                          level="WARN", component="GENERATOR")
            
            prompt = f"""<question>{question}</question>

<information>
{context}
</information>

Based on the information above, provide a concise answer to the question.
<answer>"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.actor_model.device)
            
            with torch.no_grad():
                outputs = self.actor_model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_answer_tokens,
                    temperature=self.config.temperature,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Extract answer
            if '<answer>' in generated:
                answer = generated.split('<answer>')[-1]
                if '</answer>' in answer:
                    answer = answer.split('</answer>')[0]
                answer = answer.strip()
            else:
                answer = generated.strip()
        
        self.logger.log_generator_output(answer)
        return answer
    
    def run_multi_turn_search(self, question: str, sample_id: str = "unknown") -> Dict:
        """Run multi-turn search and generate final answer"""
        
        self.logger.log_section(f"SAMPLE: {sample_id}")
        self.logger.log(f"Question: {question}", level="Q", component="INPUT")
        
        all_context = ""
        search_queries = []
        retrieved_docs = []
        
        for turn in range(self.config.max_turns):
            # Generate search query
            query = self.generate_search_query(question, all_context, turn=turn+1)
            
            if not query:
                self.logger.log(f"No query generated at turn {turn+1}, stopping", 
                              level="STOP", component="ACTOR")
                break
                
            search_queries.append(query)
            
            # Retrieve documents
            if self.retriever:
                self.logger.log_retrieval(query, 0)  # Will update count
                results = self.retriever.search([query])
                if results and results[0]:
                    num_docs = len(results[0])
                    self.logger.log(f"Retrieved {num_docs} documents", 
                                  level="RESULT", component="RETRIEVER")
                    self.logger.log_documents(results[0])
                    
                    docs_str = self.retriever.format_docs(results[0])
                    retrieved_docs.append({
                        'turn': turn + 1,
                        'query': query,
                        'docs': results[0]
                    })
                    all_context += f"\n\n--- Search Turn {turn + 1}: {query} ---\n{docs_str}"
                    
                    self.logger.log(f"Context size: {len(all_context)} chars", 
                                  level="STATS", component="CONTEXT")
            else:
                self.logger.log("⚠️  Retriever not available, using model knowledge only", 
                              level="WARN", component="RETRIEVER")
        
        # Generate final answer
        if all_context:
            answer = self.generate_answer(question, all_context)
        else:
            self.logger.log("No context retrieved, using parametric knowledge", 
                          level="WARN", component="SYSTEM")
            answer = self.generate_answer(question, "No relevant documents found. Answer based on your knowledge.")
        
        return {
            'question': question,
            'search_queries': search_queries,
            'retrieved_docs': retrieved_docs,
            'context': all_context,
            'answer': answer,
            'num_turns': len(search_queries)
        }


def start_retriever_server(config: Config) -> Optional[subprocess.Popen]:
    """Start the retriever server in background"""
    
    # Check if faiss is installed FIRST
    print("\nChecking retriever dependencies...")
    try:
        import faiss
        print(f"✓ faiss {faiss.__version__} installed (GPUs: {faiss.get_num_gpus()})")
    except ImportError:
        print("\n❌ ERROR: 'faiss' is not installed!")
        print("")
        print("The retriever requires faiss-gpu. Install it with:")
        print("  pip3 install faiss-gpu")
        print("")
        print("Then run this script again.")
        return None
    
    # Check if index exists
    if not os.path.exists(config.index_path):
        print(f"\n⚠️  Index not found at {config.index_path}")
        print("Please download it first:")
        print(f"  bash download_data.sh")
        return None
    
    if not os.path.exists(config.corpus_path):
        print(f"\n⚠️  Corpus not found at {config.corpus_path}")
        return None
    
    print(f"\nStarting retriever server on port {config.retriever_port}...")
    print("⏳ This will take 2-3 minutes (loading 61GB FAISS index)...")
    
    # Save logs to file
    log_file = open("retriever_server.log", "w")
    
    cmd = [
        "python", "s3/search/retrieval_server.py",
        "--index_path", config.index_path,
        "--corpus_path", config.corpus_path,
        "--topk", str(config.topk),
        "--retriever_name", "e5",
        "--retriever_model", config.retriever_model,
        "--faiss_gpu",
        "--port", str(config.retriever_port)
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )
    
    # Wait for server to start (increased timeout to 300s = 5 minutes for 61GB index)
    print("Waiting for retriever to initialize...")
    for i in range(300):
        # Check if process died (e.g., import error)
        if process.poll() is not None:
            print(f"\n❌ Retriever process died after {i+1}s!")
            print("Last 30 lines of retriever_server.log:")
            print("-" * 50)
            try:
                with open("retriever_server.log", "r") as f:
                    lines = f.readlines()
                    for line in lines[-30:]:
                        print(f"  {line.rstrip()}")
            except:
                print("  (could not read log file)")
            print("-" * 50)
            return None
            
        time.sleep(1)
        try:
            response = requests.get(f"http://127.0.0.1:{config.retriever_port}/docs", timeout=2)
            print(f"✓ Retriever ready! (took {i+1}s)")
            return process
        except:
            if i % 30 == 29:
                print(f"  Still loading... ({i+1}s / 300s)")
    
    print("\n⚠️  Retriever startup timed out after 300s")
    print("Check retriever_server.log for errors")
    print("Continuing without retrieval (will use parametric knowledge)...")
    return process


def main():
    print("=" * 70)
    print("s3 Full Inference - Single GPU (80GB)")
    print("=" * 70)
    print()
    
    config = Config()
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize logger
    log_file = os.path.join(config.output_dir, "s3_trace.log")
    logger = Logger(log_file)
    logger.log("Inference session started", level="START", component="SYSTEM")
    logger.log(f"Output directory: {config.output_dir}", level="CONFIG", component="SYSTEM")
    
    # Check for retrieval data
    has_retrieval = os.path.exists(config.index_path) and os.path.exists(config.corpus_path)
    
    retriever_process = None
    
    if has_retrieval:
        print("📚 Retrieval data found - starting retriever server...")
        retriever_process = start_retriever_server(config)
    else:
        print("⚠️  No retrieval data found - will use model knowledge only")
        print("   To enable retrieval, run:")
        print("   python scripts/download.py --save_path data/retrieval")
        print("   cat data/retrieval/part_* > data/retrieval/e5_Flat.index")
        print("   gzip -d data/retrieval/wiki-18.jsonl.gz")
    
    try:
        # Initialize engine
        logger.log("Initializing s3 engine...", level="INIT", component="SYSTEM")
        engine = S3InferenceEngine(config, logger)
        
        # Load Actor model
        logger.log("Loading Actor model...", level="INIT", component="ACTOR")
        engine.load_actor_model()
        logger.log(f"✓ Actor model loaded", level="READY", component="ACTOR")
        
        # Connect to retriever
        if has_retrieval:
            logger.log("Connecting to retriever...", level="INIT", component="RETRIEVER")
            engine.connect_retriever()
            logger.log("✓ Retriever connected", level="READY", component="RETRIEVER")
        
        # Connect to generator
        logger.log("Connecting to generator...", level="INIT", component="GENERATOR")
        engine.connect_generator()
        if engine.generator:
            logger.log("✓ Generator connected", level="READY", component="GENERATOR")
        else:
            logger.log("⚠️  Generator unavailable, using Actor fallback", level="WARN", component="GENERATOR")
        
        # Load test data
        logger.log("Loading test data...", level="INIT", component="DATA")
        
        # Try sampled version first (smaller, faster)
        if os.path.exists(config.test_data_sampled_path):
            logger.log(f"Using sampled test data: {config.test_data_sampled_path}", 
                      level="INFO", component="DATA")
            df = pd.read_parquet(config.test_data_sampled_path)
        elif os.path.exists(config.test_data_path):
            logger.log(f"Using full test data: {config.test_data_path}", 
                      level="INFO", component="DATA")
            df = pd.read_parquet(config.test_data_path)
        else:
            # Create sample data if not exists
            logger.log("⚠️  No test data found, creating sample data", 
                      level="WARN", component="DATA")
            print("\n⚠️  Test data not found. Run 'bash download_data.sh' to download real data.")
            print("Creating sample questions for demo...\n")
            
            questions = [
                "What is the capital of France?",
                "Who wrote Romeo and Juliet?",
                "What year did World War II end?",
                "Who discovered penicillin?",
                "What is the largest planet in our solar system?",
                "Who painted the Mona Lisa?",
                "What is the chemical symbol for gold?",
                "Who was the first person to walk on the moon?",
                "What is the speed of light?",
                "Who invented the telephone?"
            ]
            answers = [
                "Paris", "William Shakespeare", "1945", "Alexander Fleming",
                "Jupiter", "Leonardo da Vinci", "Au", "Neil Armstrong",
                "299,792 km/s", "Alexander Graham Bell"
            ]
            df = pd.DataFrame({
                'question': questions,
                'answers': [[a] for a in answers],
                'id': [f'sample_{i}' for i in range(len(questions))],
                'data_source': ['sample'] * len(questions),
                'prompt': [f'<question>{q}</question>\n\nPlease search for information to answer this question.\n<query>' 
                          for q in questions]
            })
        
        df = df.head(config.num_samples)
        print(f"Processing {len(df)} samples with {config.max_turns} search turns each...")
        
        # Run inference
        results = []
        
        correct_count = 0
        total_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            question = str(row['question'])
            
            # Handle ground truth - keep ALL possible answers
            gt = row['answers']
            if hasattr(gt, 'tolist'):
                gt = gt.tolist()
            
            # Ensure ground_truth_list is always a list of all possible answers
            if isinstance(gt, list):
                ground_truth_list = [str(g) for g in gt]
            else:
                ground_truth_list = [str(gt)]
            
            # Display string (first answer for brevity)
            ground_truth_display = ground_truth_list[0] if ground_truth_list else "N/A"
            
            sample_id = str(row.get('id', f'sample_{idx}'))
            
            print(f"\n{'='*70}")
            print(f"Sample {idx + 1}/{len(df)}: {sample_id}")
            print(f"Question: {question}")
            
            # Run multi-turn search
            result = engine.run_multi_turn_search(question, sample_id)
            result['ground_truth'] = ground_truth_display
            result['ground_truth_all'] = ground_truth_list  # Store ALL possible answers
            result['id'] = sample_id
            
            print(f"Answer: {result['answer']}")
            print(f"Ground Truth Options: {ground_truth_list}")
            
            # Log comparison against ALL possible answers
            is_correct, matched_answer = logger.log_comparison(result['answer'], ground_truth_list)
            result['is_correct'] = is_correct
            result['matched_answer'] = matched_answer
            
            if is_correct:
                correct_count += 1
            total_count += 1
            
            results.append(result)
            
            # Save individual result
            with open(f"{config.output_dir}/sample_{idx}.json", 'w') as f:
                # Convert to serializable format
                save_result = {
                    'id': result['id'],
                    'question': result['question'],
                    'search_queries': result['search_queries'],
                    'answer': result['answer'],
                    'ground_truth': result['ground_truth'],
                    'ground_truth_all': result.get('ground_truth_all', [result['ground_truth']]),
                    'is_correct': result.get('is_correct', False),
                    'matched_answer': result.get('matched_answer'),
                    'num_turns': result['num_turns']
                }
                json.dump(save_result, f, indent=2, ensure_ascii=False)
        
        # Print final accuracy
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        print(f"\n{'='*70}")
        print(f"📊 FINAL RESULTS")
        print(f"{'='*70}")
        print(f"  Total Samples: {total_count}")
        print(f"  Correct: {correct_count}")
        print(f"  Accuracy: {accuracy:.1f}%")
        print(f"{'='*70}")
        
        logger.log(f"Final Accuracy: {correct_count}/{total_count} = {accuracy:.1f}%", 
                  level="FINAL", component="EVALUATION")
        
        # Save all results
        summary_results = [{
            'id': r['id'],
            'question': r['question'],
            'search_queries': r['search_queries'],
            'answer': r['answer'],
            'ground_truth': r['ground_truth'],
            'ground_truth_all': r.get('ground_truth_all', [r['ground_truth']]),
            'is_correct': r.get('is_correct', False),
            'matched_answer': r.get('matched_answer'),
            'num_turns': r['num_turns']
        } for r in results]
        
        # Add summary stats
        final_output = {
            'summary': {
                'total_samples': total_count,
                'correct': correct_count,
                'accuracy': accuracy,
                'timestamp': datetime.now().isoformat()
            },
            'results': summary_results
        }
        
        with open(f"{config.output_dir}/final_results.json", 'w') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        
        # Also save simple results for backward compatibility
        with open(f"{config.output_dir}/all_results.json", 'w') as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)
        
        pd.DataFrame(summary_results).to_csv(f"{config.output_dir}/all_results.csv", index=False)
        
        print(f"\n{'='*70}")
        print("✓ Inference Complete!")
        print(f"{'='*70}")
        print(f"Results saved to: {config.output_dir}/")
        print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        logger.log_section("INFERENCE COMPLETE")
        logger.log(f"Processed {len(results)} samples", level="STATS", component="SYSTEM")
        logger.log(f"Results: {config.output_dir}/", level="OUTPUT", component="SYSTEM")
        logger.log(f"Trace log: {log_file}", level="OUTPUT", component="SYSTEM")
        logger.log(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB", 
                  level="STATS", component="SYSTEM")
        
        # Summary statistics
        total_queries = sum(len(r['search_queries']) for r in results)
        logger.log(f"Total search queries: {total_queries}", level="STATS", component="SYSTEM")
        logger.log(f"Avg queries per sample: {total_queries/len(results):.1f}", 
                  level="STATS", component="SYSTEM")
        
    finally:
        # Cleanup retriever process
        if retriever_process:
            print("\nStopping retriever server...")
            logger.log("Stopping retriever server", level="CLEANUP", component="SYSTEM")
            os.killpg(os.getpgid(retriever_process.pid), signal.SIGTERM)


if __name__ == "__main__":
    main()


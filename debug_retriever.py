#!/usr/bin/env python3
"""
Debug script to test retriever loading step by step
"""

import os
import sys
import time

# Add project to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("=" * 60)
print("Retriever Debug Script")
print("=" * 60)
print()

# Step 1: Check faiss
print("STEP 1: Testing faiss import...")
start = time.time()
try:
    import faiss
    print(f"  ✓ faiss {faiss.__version__} imported in {time.time()-start:.1f}s")
    print(f"  ✓ GPUs available: {faiss.get_num_gpus()}")
except Exception as e:
    print(f"  ❌ faiss import failed: {e}")
    sys.exit(1)

# Step 2: Check torch/CUDA
print()
print("STEP 2: Testing torch/CUDA...")
start = time.time()
try:
    import torch
    print(f"  ✓ torch {torch.__version__} imported")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  ✓ GPU Memory Used: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
except Exception as e:
    print(f"  ❌ torch/CUDA failed: {e}")

# Step 3: Check index file
print()
print("STEP 3: Checking index file...")
index_path = "data/retrieval/e5_Flat.index"
if os.path.exists(index_path):
    size_gb = os.path.getsize(index_path) / 1e9
    print(f"  ✓ Index exists: {index_path}")
    print(f"  ✓ Size: {size_gb:.1f} GB")
else:
    print(f"  ❌ Index not found: {index_path}")
    sys.exit(1)

# Step 4: Check corpus file
print()
print("STEP 4: Checking corpus file...")
corpus_path = "data/retrieval/wiki-18.jsonl"
if os.path.exists(corpus_path):
    size_gb = os.path.getsize(corpus_path) / 1e9
    print(f"  ✓ Corpus exists: {corpus_path}")
    print(f"  ✓ Size: {size_gb:.1f} GB")
else:
    print(f"  ❌ Corpus not found: {corpus_path}")
    sys.exit(1)

# Step 5: Load FAISS index (CPU only first)
print()
print("STEP 5: Loading FAISS index (CPU only)...")
print("  ⏳ This may take 2-5 minutes for 61GB index...")
start = time.time()
try:
    index = faiss.read_index(index_path)
    elapsed = time.time() - start
    print(f"  ✓ Index loaded in {elapsed:.1f}s")
    print(f"  ✓ Index size: {index.ntotal} vectors")
    print(f"  ✓ Index dimension: {index.d}")
except Exception as e:
    print(f"  ❌ Index loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Check GPU memory before transfer
print()
print("STEP 6: GPU memory check before index transfer...")
if torch.cuda.is_available():
    free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    print(f"  ✓ Free GPU memory: {free_mem / 1e9:.1f} GB")
    print(f"  ✓ Index size estimate: ~{index.ntotal * index.d * 4 / 1e9:.1f} GB (float32)")
    
    # Estimate if it will fit
    index_size_estimate = index.ntotal * index.d * 4  # float32
    if index_size_estimate > free_mem * 0.9:
        print(f"  ⚠️  WARNING: Index may not fit in GPU memory!")
        print(f"      Index needs: ~{index_size_estimate / 1e9:.1f} GB")
        print(f"      GPU has free: {free_mem / 1e9:.1f} GB")
        print()
        print("  Options:")
        print("    1. Skip GPU transfer (use CPU for retrieval)")
        print("    2. Continue anyway (may crash)")
        print()
        response = input("  Try GPU transfer anyway? (y/n): ").strip().lower()
        if response != 'y':
            print()
            print("  Skipping GPU transfer - will use CPU retrieval")
            USE_GPU = False
        else:
            USE_GPU = True
    else:
        USE_GPU = True
else:
    USE_GPU = False
    print("  ⚠️  No CUDA available, will use CPU retrieval")

# Step 7: Transfer to GPU (if applicable)
if USE_GPU and torch.cuda.is_available() and faiss.get_num_gpus() > 0:
    print()
    print("STEP 7: Transferring index to GPU...")
    print("  ⏳ This may take 1-2 minutes...")
    start = time.time()
    try:
        gpu_index = faiss.index_cpu_to_all_gpus(index)
        elapsed = time.time() - start
        print(f"  ✓ Index transferred to GPU in {elapsed:.1f}s")
        print(f"  ✓ GPU Memory Used: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        index = gpu_index
    except Exception as e:
        print(f"  ❌ GPU transfer failed: {e}")
        print("  Falling back to CPU retrieval")
        import traceback
        traceback.print_exc()
else:
    print()
    print("STEP 7: Skipping GPU transfer (using CPU)")

# Step 8: Load corpus
print()
print("STEP 8: Loading corpus...")
print("  ⏳ This may take 1-2 minutes for 14GB corpus...")
start = time.time()
try:
    import datasets
    corpus = datasets.load_dataset(
        'json',
        data_files=corpus_path,
        split="train",
        num_proc=4
    )
    elapsed = time.time() - start
    print(f"  ✓ Corpus loaded in {elapsed:.1f}s")
    print(f"  ✓ Corpus size: {len(corpus)} documents")
except Exception as e:
    print(f"  ❌ Corpus loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 9: Load encoder model
print()
print("STEP 9: Loading E5 encoder model...")
start = time.time()
try:
    from transformers import AutoModel, AutoTokenizer
    encoder_path = "intfloat/e5-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    model = AutoModel.from_pretrained(encoder_path)
    model.eval()
    model.cuda()
    elapsed = time.time() - start
    print(f"  ✓ Encoder loaded in {elapsed:.1f}s")
    print(f"  ✓ GPU Memory Used: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
except Exception as e:
    print(f"  ❌ Encoder loading failed: {e}")
    import traceback
    traceback.print_exc()

# Step 10: Test a simple search
print()
print("STEP 10: Testing search...")
try:
    import numpy as np
    
    # Encode a test query
    test_query = "who invented the telephone"
    inputs = tokenizer(f"query: {test_query}", return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        query_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    
    # Search
    start = time.time()
    D, I = index.search(query_emb.astype('float32'), 5)
    elapsed = time.time() - start
    
    print(f"  ✓ Search completed in {elapsed*1000:.1f}ms")
    print(f"  ✓ Top 5 doc indices: {I[0].tolist()}")
    print(f"  ✓ Top 5 scores: {D[0].tolist()}")
    
    # Get document
    if len(I[0]) > 0 and I[0][0] >= 0:
        doc_idx = int(I[0][0])
        doc = corpus[doc_idx]
        print(f"  ✓ Top result preview: {doc['contents'][:200]}...")

except Exception as e:
    print(f"  ❌ Search test failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("Debug Complete!")
print("=" * 60)
print()
print("Summary:")
print(f"  - FAISS index: {index.ntotal} vectors")
print(f"  - Corpus: {len(corpus)} documents")
print(f"  - GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
print()

if torch.cuda.is_available():
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    used_mem = torch.cuda.memory_allocated(0) / 1e9
    print(f"GPU Usage: {used_mem:.1f} / {total_mem:.1f} GB ({used_mem/total_mem*100:.1f}%)")


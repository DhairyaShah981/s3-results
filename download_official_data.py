#!/usr/bin/env python3
"""
Download official s3 processed data from HuggingFace.
Source: https://huggingface.co/datasets/pat-jj/s3_processed_data

Files available:
- train_e5_s3.parquet (334 MB) - Training data with E5 embeddings
- test_e5_s3.parquet (195 MB) - Test data with E5 embeddings  
- mirage_wiki_test.parquet (14.2 MB) - MIRAGE Wikipedia test data
- mirage_medcorp_test.parquet (20.2 MB) - MIRAGE Medical corpus test data
"""

import os
import argparse
from huggingface_hub import hf_hub_download

def download_s3_data(save_path: str, files: list = None):
    """Download s3 processed data from HuggingFace"""
    
    repo_id = "pat-jj/s3_processed_data"
    
    # Default: download all files
    if files is None:
        files = [
            "train_e5_s3.parquet",
            "test_e5_s3.parquet",
            "mirage_wiki_test.parquet",
            "mirage_medcorp_test.parquet"
        ]
    
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Downloading s3 processed data to: {save_path}")
    print(f"Source: https://huggingface.co/datasets/{repo_id}")
    print("-" * 60)
    
    for file in files:
        print(f"\nDownloading: {file}")
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=file,
                repo_type="dataset",
                local_dir=save_path,
            )
            print(f"  ✓ Saved to: {local_path}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✓ Download complete!")
    print(f"Files saved to: {save_path}")


def download_retrieval_data(save_path: str):
    """Download retrieval index and corpus (~60GB total)"""
    
    print("\n" + "=" * 60)
    print("Downloading retrieval data (Index + Corpus)...")
    print("⚠️  This will download ~60GB of data")
    print("=" * 60)
    
    os.makedirs(save_path, exist_ok=True)
    
    # Download index parts
    print("\n[1/3] Downloading FAISS index parts...")
    index_repo = "PeterJinGo/wiki-18-e5-index"
    for file in ["part_aa", "part_ab"]:
        print(f"  Downloading {file}...")
        hf_hub_download(
            repo_id=index_repo,
            filename=file,
            repo_type="dataset",
            local_dir=save_path,
        )
    
    # Combine index parts
    print("\n[2/3] Combining index parts...")
    index_path = os.path.join(save_path, "e5_Flat.index")
    if not os.path.exists(index_path):
        os.system(f"cat {save_path}/part_aa {save_path}/part_ab > {index_path}")
        print(f"  ✓ Created: {index_path}")
    else:
        print(f"  ⏭️  Index already exists: {index_path}")
    
    # Download corpus
    print("\n[3/3] Downloading Wikipedia corpus...")
    corpus_repo = "PeterJinGo/wiki-18-corpus"
    hf_hub_download(
        repo_id=corpus_repo,
        filename="wiki-18.jsonl.gz",
        repo_type="dataset",
        local_dir=save_path,
    )
    
    # Decompress corpus
    corpus_gz = os.path.join(save_path, "wiki-18.jsonl.gz")
    corpus_path = os.path.join(save_path, "wiki-18.jsonl")
    if not os.path.exists(corpus_path):
        print("  Decompressing corpus...")
        os.system(f"gzip -dk {corpus_gz}")
        print(f"  ✓ Created: {corpus_path}")
    else:
        print(f"  ⏭️  Corpus already exists: {corpus_path}")
    
    print("\n" + "=" * 60)
    print("✓ Retrieval data download complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download s3 data from HuggingFace")
    parser.add_argument("--save_path", type=str, default="data/s3_official",
                       help="Directory to save processed data")
    parser.add_argument("--retrieval_path", type=str, default="data/retrieval",
                       help="Directory to save retrieval data (index + corpus)")
    parser.add_argument("--download_retrieval", action="store_true",
                       help="Also download retrieval data (~60GB)")
    parser.add_argument("--test_only", action="store_true",
                       help="Only download test data (smaller)")
    
    args = parser.parse_args()
    
    # Download processed data
    if args.test_only:
        files = ["test_e5_s3.parquet"]
    else:
        files = None  # Download all
    
    download_s3_data(args.save_path, files)
    
    # Optionally download retrieval data
    if args.download_retrieval:
        download_retrieval_data(args.retrieval_path)
    else:
        print("\n💡 To also download retrieval data (index + corpus), run:")
        print(f"   python {__file__} --download_retrieval")


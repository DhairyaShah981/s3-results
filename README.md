
<div align="right">
  <details>
    <summary >🌐 Language</summary>
    <div>
      <div align="center">
        <a href="https://openaitx.github.io/view.html?user=pat-jj&project=s3&lang=en">English</a>
        | <a href="https://openaitx.github.io/view.html?user=pat-jj&project=s3&lang=zh-CN">简体中文</a>
        | <a href="https://openaitx.github.io/view.html?user=pat-jj&project=s3&lang=zh-TW">繁體中文</a>
        | <a href="https://openaitx.github.io/view.html?user=pat-jj&project=s3&lang=ja">日本語</a>
        | <a href="https://openaitx.github.io/view.html?user=pat-jj&project=s3&lang=ko">한국어</a>
        | <a href="https://openaitx.github.io/view.html?user=pat-jj&project=s3&lang=hi">हिन्दी</a>
        | <a href="https://openaitx.github.io/view.html?user=pat-jj&project=s3&lang=th">ไทย</a>
        | <a href="https://openaitx.github.io/view.html?user=pat-jj&project=s3&lang=fr">Français</a>
        | <a href="https://openaitx.github.io/view.html?user=pat-jj&project=s3&lang=de">Deutsch</a>
        | <a href="https://openaitx.github.io/view.html?user=pat-jj&project=s3&lang=es">Español</a>
        | <a href="https://openaitx.github.io/view.html?user=pat-jj&project=s3&lang=it">Itapano</a>
        | <a href="https://openaitx.github.io/view.html?user=pat-jj&project=s3&lang=ru">Русский</a>
        | <a href="https://openaitx.github.io/view.html?user=pat-jj&project=s3&lang=pt">Português</a>
        | <a href="https://openaitx.github.io/view.html?user=pat-jj&project=s3&lang=nl">Nederlands</a>
        | <a href="https://openaitx.github.io/view.html?user=pat-jj&project=s3&lang=pl">Polski</a>
        | <a href="https://openaitx.github.io/view.html?user=pat-jj&project=s3&lang=ar">العربية</a>
        | <a href="https://openaitx.github.io/view.html?user=pat-jj&project=s3&lang=fa">فارسی</a>
        | <a href="https://openaitx.github.io/view.html?user=pat-jj&project=s3&lang=tr">Türkçe</a>
        | <a href="https://openaitx.github.io/view.html?user=pat-jj&project=s3&lang=vi">Tiếng Việt</a>
        | <a href="https://openaitx.github.io/view.html?user=pat-jj&project=s3&lang=id">Bahasa Indonesia</a>
      </div>
    </div>
  </details>
</div>

<div align="center">

# s3 - Efficient Yet Effective Search Agent Training via RL
***You Don't Need That Much Data to Train a Search Agent***

<p align="center">

  <a href="https://arxiv.org/abs/2505.14146">
    <img src="https://img.shields.io/badge/arXiv-2505.14146-b31b1b.svg" alt="arXiv">
  </a>
</p>
</div>

**Performance Overview:**

<img src="images/performance_overview.png" alt="performance_overview" width="800">



## What is s3?

<div align="center">
<img src="images/framework.png" alt="framework" width="800">

**s3 Framework**
</div>

`s3` is a simple yet powerful framework for training search agents in retrieval-augmented generation (RAG). It teaches language models how to search more effectively—without changing the generator itself. By focusing solely on the search component, `s3` achieves strong performance across QA tasks with just a fraction of the data used by prior methods. It's modular, efficient, and designed to work seamlessly with any black-box LLM.



## Table of Contents

- [📦 Installation](#-installation)
- [💡 Preparation](#-preparation)
- [🏋️ Run Training](https://github.com/pat-jj/s3?tab=readme-ov-file#%EF%B8%8F-run-training)
- [🔍 Run Search/Retrieval](https://github.com/pat-jj/s3?tab=readme-ov-file#-run-searchretrieval)
- [📈 Run Evaluation](#-run-evaluation)

## 📦 Installation

**Searcher & Generator Environment**
```bash
conda create -n s3 python=3.9
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray

# verl
# cd code
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation

# we use pyserini for efficient retrieval and evaluation
pip install pyserini    # the version we used is 0.22.1

# quality of life
pip install wandb IPython matplotlib huggingface_hub
```

**Retriever Environment**
```bash
conda create -n ret python=3.10
conda activate ret

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install uvicorn fastapi
```



## 💡 Preparation
***Download Index & Corpus***
```bash
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

***Precompute Naïve RAG Initialization*** (or you can download our processed data here: [huggingface](https://huggingface.co/datasets/pat-jj/s3_processed_data))

```bash
# deploy retriever
bash scripts/deploy_retriever/retrieval_launch.sh # or scripts/deploy_retriever/retrieval_launch_mirage.sh for MedCorp corpus.
# deploy generator
bash generator_llms/host.sh # modify tensor-parallel-size to the number of GPUs you use
# run precompute
bash scripts/precompute.sh # this step will take a while, as it will precompute the naïve RAG Cache for training
```


## 🏋️ Run Training
***This step is for the training of S3***

```bash
# deploy retriever
bash scripts/deploy_retriever/retrieval_launch.sh 
# deploy generator
bash generator_llms/host.sh
# run training
bash scripts/train/train_s3.sh
```


## 🔍 Run Search/Retrieval
***This step is for the context gathering of s3 / baselines***

**s3**
```bash
# deploy retriever
bash scripts/deploy_retriever/retrieval_launch.sh 
# run s3 inference
bash scripts/s3_inference/evaluate-8-3-3.sh
```

<details>
<summary>Baselines</summary>

**RAG**
```bash
bash scripts/deploy_retriever/retrieval_launch.sh # or retrieval_launch_bm25.sh # deploy retriever
bash scripts/baselines/rag.sh # run RAG 
```

**DeepRetrieval**
```bash
bash retrieval_launch_bm25.sh # deploy BM25 Model
bash generator_llms/deepretrieval.sh # deploy DeepRetrieval Model
bash scripts/baselines/deepretrieval.sh # run DeepRetrieval Query Rewriting + Retrieval
```

**Search-R1**
```bash
bash retrieval_launch.sh # deploy e5 retriever
bash scripts/baselines/search_r1.sh # run Search-R1
```

**IRCoT**
```bash
bash retrieval_launch.sh # deploy e5 retriever
python scripts/baselines/ircot.py
```

**Search-o1**
```bash
bash retrieval_launch.sh # deploy e5 retriever
bash scripts/baselines/search_o1.sh # run Search-o1
```

</details>


## 📈 Run Evaluation
***This step is for the evaluation of s3 / baselines***


```bash
bash scripts/evaluation/run.sh
```

## Q&A
### Customized Data?
If you want to test s3 on your own corpus/dataset, you can refer to this commit to see what you need to do to build your own pipeline: [commit 8420538](https://github.com/pat-jj/s3/commit/8420538836febbe59d5bcbe41187f16908c9c36c)

### Reproducing Results?
Several developers have already reproduced our results successfully. If you have questions or run into issues, feel free to [open an issue](https://github.com/pat-jj/s3/issues) — we’re happy to provide hands-on guidance (see [this example](https://github.com/pat-jj/s3/issues/20)).

Although reproducing the model yourself is straightforward — and we actually **recommend training from scratch**, since evaluation is often much more time-consuming than training — we also provide a reference checkpoint: [s3-8-3-3-20steps](https://huggingface.co/pat-jj/s3-8-3-3-20steps), trained in about one hour.



---

## 🔬 Dhairya's Reproduction Effort

This section documents the successful reproduction of S3 inference results on RunPod infrastructure.

### 📊 Reproduction Results (Quick Test - 10 Samples)

| Dataset | Samples | Accuracy | Exact Match |
|---------|---------|----------|-------------|
| triviaqa | 1 | 100% | 0% |
| 2wikimultihopqa | 1 | 100% | 100% |
| popqa | 2 | 100% | 0% |
| hotpotqa | 2 | 0% | 0% |
| musique | 1 | 0% | 0% |
| nq | 3 | 100% | 33% |
| **OVERALL** | **10** | **70%** | **20%** |

*Metrics match paper's evaluation: Accuracy = span check + LLM semantic check; EM = normalized string equality*

### 🖥️ Infrastructure Used

- **Platform**: RunPod
- **GPUs**: 2× NVIDIA A100 SXM 80GB
- **Container Disk**: 100GB
- **Volume Disk**: 150GB
- **Base Image**: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`

### 🚀 Quick Start (RunPod Reproduction)

```bash
# 1. Setup environment (run once)
bash reproduction/scripts/RUNPOD_SETUP.sh

# 2. Start servers (Retriever + Generator)
bash reproduction/scripts/start_servers.sh

# 3. Run quick test (10 samples)
bash reproduction/scripts/run_quick_test.sh

# 4. Evaluate results
python3 reproduction/scripts/evaluate_with_paper_metrics.py \
    --input_dir data/output_quick_test
```

### 📁 Reproduction Files

```
reproduction/
├── scripts/
│   ├── RUNPOD_SETUP.sh           # Environment setup
│   ├── start_servers.sh          # Start Retriever + Generator
│   ├── run_quick_test.sh         # Run inference (10 samples)
│   └── evaluate_with_paper_metrics.py  # Evaluation script
├── results/
│   ├── evaluation_results_paper_metrics.json
│   └── output_quick_test/        # JSON output sequences
└── logs/
    └── quick_test_trace.log      # Detailed inference trace
```

### 🔧 Key Modifications Made

1. **`s3/search/retrieval_server.py`**: Fixed FAISS GPU allocation to prevent OOM
2. **`verl/trainer/main_ppo.py`**: Added numpy-to-list conversion for JSON serialization
3. **`generator_llms/gpt_azure.py`**: Updated langchain imports for compatibility

### 📋 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        S3 Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌───────────┐    ┌───────────┐            │
│  │  Actor   │───►│ Retriever │───►│ Generator │            │
│  │(Search   │    │(E5 + FAISS│    │(Qwen 7B   │            │
│  │ Agent)   │    │  Index)   │    │ GPTQ-Int4)│            │
│  └──────────┘    └───────────┘    └───────────┘            │
│       │                │                │                   │
│       │ GPU 0+1        │ GPU 1          │ GPU 0             │
│       │ ~15GB          │ ~24GB          │ ~15GB             │
│       ▼                ▼                ▼                   │
│   Query Gen      Doc Retrieval    Answer Gen               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 🐛 Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| CUDA OOM | Use `gpu_memory_utilization=0.4`, separate GPUs for components |
| SSH disconnect | Add `ServerAliveInterval 60` to SSH config |
| pyairports import error | Patch outlines types/__init__.py |
| numpy JSON serialization | Convert arrays to lists before json.dump |

### 👤 Author

**Dhairya Shah** - [GitHub](https://github.com/DhairyaShah981)

Reproduction completed: December 2024

---

## Citation
```bibtex
@article{jiang2025s3,
  title={s3: You Don't Need That Much Data to Train a Search Agent via RL},
  author={Jiang, Pengcheng and Xu, Xueqiang and Lin, Jiacheng and Xiao, Jinfeng and Wang, Zifeng and Sun, Jimeng and Han, Jiawei},
  journal={arXiv preprint arXiv:2505.14146},
  year={2025}
}
```

Thanks for your interest in our work!




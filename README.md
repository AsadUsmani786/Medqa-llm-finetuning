# Medical Q&A Fine-tuning with Mistral-7B + QLoRA

**Author:** Asad Usmani  
**GitHub:** [@AsadUsmani786](https://github.com/AsadUsmani786)  
**Model:** [HuggingFace - Asad854/mistral-medical-qlora](https://huggingface.co/Asad854/mistral-medical-qlora)

## Overview

This project fine-tunes Mistral-7B on the MedMCQA dataset using QLoRA (Quantized Low-Rank Adaptation) for parameter-efficient domain adaptation. The goal is to improve the model's performance on medical multiple-choice questions while keeping compute costs minimal.

## Results

| Model | Accuracy (300-question eval) |
|-------|------------------------------|
| Mistral-7B Base (no fine-tuning) | 41.0% |
| Mistral-7B + QLoRA Fine-tuned | 52.7% |
| **Improvement** | **+11.7%** |

Trained on 8,000 samples for 1 epoch on a single T4 GPU (free Kaggle tier).

## Dataset

**MedMCQA** — 194k multiple-choice questions from Indian medical entrance exams (AIIMS, NEET PG) covering 21 medical subjects.

- Training samples used: 8,000
- Validation samples: 500
- Evaluation set: 300 held-out questions

## Method

### Why QLoRA?
Full fine-tuning of a 7B model requires ~80GB VRAM and thousands of dollars. QLoRA makes it possible on a single consumer GPU by:
1. Loading the model in **4-bit quantization** (reduces memory from ~14GB to ~5GB)
2. Adding small **LoRA adapter matrices** to the attention layers
3. Training only the adapters while keeping original weights frozen

### Training Configuration

```
Model:          mistralai/Mistral-7B-v0.1
LoRA rank:      r=32, alpha=64
Target modules: q_proj, k_proj, v_proj, o_proj
Epochs:         1
Batch size:     4 (with 4 gradient accumulation steps)
Learning rate:  2e-4
Max length:     512 tokens
Hardware:       Kaggle T4 GPU (16GB VRAM)
Training time:  ~5 hours 18 minutes
```



## How to Run

1. Open `notebook.ipynb` on Kaggle
2. Enable GPU T4 x2 under Session options
3. Add `HF_TOKEN` and `WANDB_API_KEY` to Kaggle Secrets
4. Run all cells in order

## Requirements

```
transformers
peft
trl
bitsandbytes
datasets
wandb
```

## Key Findings

- QLoRA fine-tuning on 8,000 samples produced an 11.7 percentage point accuracy improvement over the base model
- Training and validation loss remained closely aligned (1.026 vs 1.130), indicating no overfitting
- Mean token accuracy reached ~75.8% on both train and validation sets by the end of training
- Expanding LoRA target modules (q/k/v/o vs. q/v only) and increasing rank (32 vs. 16) contributed to a stronger result than an earlier pilot run on 3,000 samples
- Full training completed in ~5.3 hours on a free T4 GPU


```

## References

- [QLoRA Paper](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [MedMCQA Dataset](https://huggingface.co/datasets/medmcqa) — Pal et al., 2022
- [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) — Mistral AI, 2023
- [PEFT Library](https://github.com/huggingface/peft) — HuggingFace

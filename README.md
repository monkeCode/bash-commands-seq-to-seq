# Bash Command Generation Model

A fine-tuned T5-small model for generating bash commands from natural language descriptions, with a comprehensive data annotation pipeline and training framework.

## Overview

This project implements a machine learning system that translates natural language requests into executable bash commands. The model is based on the T5-small architecture and has been fine-tuned on a carefully curated dataset of bash commands paired with their natural language descriptions.

## Model Architecture

- **Base Model**: T5-small (60M parameters)
- **Task**: Sequence-to-sequence translation (natural language → bash command)
- **Framework**: PyTorch Lightning with Hugging Face Transformers
- **Training Strategy**: Fine-tuning with early stopping and best model checkpointing

## Dataset

### Data Collection and Annotation

The dataset comprises over 60k real terminal commands collected from `.bash_history`, `.zsh_history` files, internet sources, and parsed man pages.

#### LLM Annotation Performance

For data annotation and filtering tasks, multiple LLMs were evaluated on a hand-annotated dataset of 100 examples. Each model classified bash commands with `is_command` labels (true/false).

| Model                              | Size | Quantization | Precision | Recall | F1    | Inference Time 100 examples (RTX 3060 Ti) |
| ---------------------------------- | ---- | ------------ | --------- | ------ | ----- | ----------------------------------------- |
| qwen/qwen3-coder-30b               | 30B  | Q4_K_M       | 0.971     | 0.827  | 0.893 | 13.0 min                                  |
| microsoft/phi-4                    | 15B  | Q4_K_M       | 0.932     | 0.85   | 0.889 | 22.2min                                   |
| qwen/qwen3-4b-2507                 | 4B   | Q4_K_M       | 0.922     | 0.728  | 0.814 | 2.3min                                    |
| mistralai/mistral-7b-instruct-v0.3 | 7B   | Q4_K_M       | 0.904     | 0.815  | 0.857 | 2.7min                                    |
| google/gemma-3n-e4b                | 6.9B | Q4_K_M       | 0.876     | 0.963  | 0.918 | 3.3min                                    |
| liquid/lfm2-1.2b                   | 1.2B | Q8_0         | 0.824     | 0.763  | 0.792 | 1.2min                                    |

Qwen3-coder-30b was selected as the primary annotation model due to its superior performance and annotation capabilities. \
The final annotated dataset contains 45,119 valid commands (70% of the original data).

### Additional Data Sources

- **Public Datasets**:
  - magnumresearchgroup/bash_gen
  - darkknight25/Linux_Terminal_Commands_Dataset
  - aelhalili/bash-commands-dataset
  - TellinaTool/nl2bash
- **Linux Man Pages**: Automated parsing of command documentation

### Final Dataset Composition

| Split    | Samples | Description                     |
| -------- | ------- | ------------------------------- |
| Training | 111,235 | Filtered and annotated commands |
| Test     | 1,190   | Manually verified commands      |

#### Command Distribution

Training commands distribution:

![train distribution](docs/train_commands_dist.png)

Test commands distribution:

![test distribution](docs/test_commands_dist.png)

## Training Setup

### Hyperparameters

| Parameter         | Value                |
| ----------------- | -------------------- |
| Base Model        | t5-small             |
| Learning Rate     | 1e-4                 |
| Batch Size        | 16                   |
| Max Epochs        | 15                   |
| Max Source Length | 128 tokens           |
| Max Target Length | 64 tokens            |
| Early Stopping    | Based on BLEU² score |
| Optimizer         | Adam                 |

### Model Performance

| Metric          | Value   |
| --------------- | ------- |
| Perplexity      | 1.21    |
| BLEU² Score     | 0.56    |
| BLEU⁴ Score     | 0.44    |
| Evaluation time | 7.3 min |

## Comparative Evaluation

### LLM Performance on Command Generation

| Model                              | Size | Quantization | BLEU² | BLEU⁴ | Inference Time 1190 examples (RTX 3060 Ti) |
| ---------------------------------- | ---- | ------------ | ----- | ----- | ------------------------------------------ |
| qwen/qwen3-coder-30b               | 30B  | Q4_K_M       | 0.338 | 0.193 | 72 min                                     |
| microsoft/phi-4                    | 15B  | Q4_K_M       | 0.293 | 0.168 | 54.9 min                                   |
| qwen/qwen3-4b-2507                 | 4B   | Q4_K_M       | 0.263 | 0.148 | 23.6 min                                   |
| google/gemma-3n-e4b                | 6.9B | Q4_K_M       | 0.277 | 0.156 | 18.8 min                                   |
| liquid/lfm2-1.2b                   | 1.2B | Q8_0         | 0.135 | 0.064 | 12.3                                       |
| mistralai/mistral-7b-instruct-v0.3 | 7B   | Q4_K_M       | 0.031 | 0.015 | 78 min                                     |

## Infrastructure

### ML Pipeline

- **Data Versioning**: DVC for dataset and pipeline management
- **Experiment Tracking**: MLflow for logging parameters, metrics, and artifacts
- **Annotation Pipeline**: LangChain for LLM-powered data annotation
- **Manual Annotation**: Flask web application for human verification
- **Training Framework**: PyTorch Lightning with Hugging Face Transformers

## Results and Deployment

The final model is available on Hugging Face: [GeraniumCat/bash-seq-to-seq](https://huggingface.co/GeraniumCat/bash-seq-to-seq)

**Example Usage**:

```python
from transformers import pipeline

pipe = pipeline("translation", model="GeraniumCat/bash-seq-to-seq")
pipe("find all files with txt extension")
```

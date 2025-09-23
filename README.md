# Bash Command Generation Model

A fine-tuned T5-small model for generating bash commands from natural language descriptions, with a comprehensive data annotation pipeline and training framework.

## Overview

This project implements a machine learning system that translates natural language requests into executable bash commands. The model is based on T5-small architecture, fine-tuned on a carefully curated dataset of bash commands and their natural language descriptions.

## Model Architecture

- **Base Model**: T5-small
- **Task**: Sequence-to-sequence translation (text → bash command)
- **Framework**: PyTorch Lightning with Hugging Face Transformers
- **Training Strategy**: Fine-tuning with early stopping and best model checkpointing

## Dataset

### Data Collection and Annotation

We collected and annotated approximately 59k real terminal commands using a multi-stage process:

#### **Initial Annotation with LFM2 1.2B**

Used LFM2 1.2B model for its superior prompt-following capabilities

<details>
<summary>Annotation prompt</summary>

```md
You are a bash command explanation system. Analyze the given bash command and provide a clear, accurate description in action format.

## Input

- Receive a bash command string
- Commands may include flags, arguments, and complex syntax

## Output Requirements

- Provide a JSON object with exactly two fields: "reasoning" and "description"
- "reasoning": Detailed technical analysis of the command, explaining each component (flags, arguments, syntax)
- "description": Concise, human-readable action of what the command does (1-2 sentences), always note paths, urls, ips, and important content
- Use English only for all output
- Check command is valid, exists and could be executed and contains no excess text or anything else, mark it in is_command field as true, false otherwise, follow the next instuctions:

### MARK is_command true if:

- command is valid and exists
- command could be executed without errors
- command not contains excess text or something else

### MARK is_command false if:

- command does not exists
- command is'n complete or contains excess text

IF COMMAND CONTAINS FILES OR IP ADDRESSES LET IT CORRECT AND EXISTS, VERFY ONLY SYNTAXIS IN COMMANDS

## Examples

Input: "cd /var/log"
Output: {{
  "reasoning": "Command 'cd' (change directory) with argument '/var/log' (absolute path to directory)",
  "description": "Change current working directory to /var/log",
  is_command:true
  }}

Input: "tar -czvf archive.tar.gz /home/user"
Output: {{
  "reasoning": "Command 'tar' (tape archive) with flags: -c (create archive), -z (gzip compression), -v (verbose output), -f (specify filename). Arguments: 'archive.tar.gz' (output filename), '/home/user' (directory to archive)",
  "description": "Create a gzip-compressed tar archive of /home/user directory",
  is_command: true
  }}

Input: "//192.168.2.101/ShareForVMs /media/share/ cifs username=toeknee,password=2dog$hit3"
Output: {{
  "reasoning": "No commands provided, it could be line in log file or part of arguments. command is incorrect",
  "description": "no command is provided",
  is_command: false
  }}

Input: "git clone git://projects.archlinux.org/archiso.git && cd archiso"
Output:{{
  "reasoning": "The command uses `git clone` to create a copy of the repository from 'git://projects.archlinux.org/archiso.git'. It then changes the current working directory to 'archiso' using `cd`. No files or IP addresses are involved.",
  "description": "Clones the Git repository from the projects.archlinux.org/archiso.git and changes the current working directory to archiso.",
  "is_command": true
  }}

## Security Note

- Do not execute or simulate execution of any commands
- Provide analysis based on command syntax and documentation only
- Flag potentially destructive commands in your reasoning

## Output Format

{format_instructions}

YOU SHOULD ALWAYS FOLLOW OUTPUT FORMAT, NO SKIPS, NO OTHER FIELDS, NO ANYTHING ELSE, ONLY reasoning, description and is_command JSON

Describe this command: {command}
```

</details>

### **Manual Validation**

Developed a Flask application for human annotation
Annotated 100 samples with three quality categories:

- **Good**: Complete correspondence between command and description
- **Partly Good**: Missing file paths or specific mechanisms but generally correct
- **Bad**: Incorrect correspondence or serious errors in file paths

### **Quality Metrics**

| Category    | Count | Percentage |
| ----------- | ----- | ---------- |
| Good        | 63    | 63%        |
| Partly Good | 12    | 12%        |
| Bad         | 10    | 10%        |

### **Annotation Performance**

Model annotated every bash line with the `is_command` true or false label. There is metrics of this annotations:

- Precision: 0.9529
- Recall: 0.92045
- F1-score: 0.9364

### **Additional Data Sources**

- Public datasets
  - <https://github.com/magnumresearchgroup/bash_gen>
  - darkknight25/Linux_Terminal_Commands_Dataset
  - aelhalili/bash-commands-dataset
- Parsing linux man pages

### Final Dataset Composition

| Split    | Samples | Description                     |
| -------- | ------- | ------------------------------- |
| Training | 93824   | Filtered and annotated commands |
| Test     | 1500    | Manually verified commands      |

## Training Setup

### Hyperparameters

| Parameter         | Value               |
| ----------------- | ------------------- |
| Base Model        | t5-small            |
| Learning Rate     | 1e-4                |
| Batch Size        | 16                  |
| Max Epochs        | 15                  |
| Max Source Length | 128 tokens          |
| Max Target Length | 64 tokens           |
| Early Stopping    | Based on BLEU score |
| Optimizer         | Adam                |

### Performance Metrics on test set

| Metric         | Value |
| -------------- | ----- |
| Perplexity     | 1.17  |
| $BLEU^2$ Score | 0.66  |

## Infrastructure

### ML Pipeline

- **Data Versioning**: DVC for dataset and pipeline management
- **Experiment Tracking**: MLflow for logging parameters, metrics, and artifacts
- **Annotation Pipeline**: LangChain for LLM-powered data annotation
- **Manual Annotation**: Flask web application for human verification
- **Training Framework**: PyTorch Lightning with Hugging Face Transformers

## Results and Evaluation

The final model was posted on hugging face and is available at the link:
example of inference:

```python
from transformers import pipeline

pipe = pipeline("translation", model="GeraniumCat/bash-seq-to-seq")
pipe("find all files with txt extension")
```

## Future Work

- Expand dataset with more diverse command patterns
- Incorporate syntax-aware decoding for improved command validity
- Develop a safety module to prevent generation of harmful commands

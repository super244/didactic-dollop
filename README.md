# Calculus Tutor (Tuning Station)

This repository contains an end-to-end pipeline for fine-tuning and evaluating a language model (default: `Qwen/Qwen2.5-0.5B-Instruct`) for calculus tutoring.

## Setup

Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you plan to use a gated model, authenticate with Hugging Face:

```bash
huggingface-cli login
# or export HF_TOKEN=your_token_here
```

## End-to-End Flow

### Step 1: Data Prep

Generate or refresh the dataset:

```bash
python dataset-generator.py
```
This creates `data/calculus_problems.jsonl`.

### Step 2: Train (Tuning Station)

Train the model using LoRA/QLoRA via the new modular trainer.

```bash
python scripts/trainer.py --use_qlora
```

**Key Features:**
- Uses PEFT (LoRA) to train adapters efficiently.
- Supports 4-bit QLoRA with the `--use_qlora` flag (requires GPU).
- Automated logging and checkpoints are saved to the output directory.
- Run `python scripts/trainer.py --help` for configuration options (e.g., `--batch_size`, `--learning_rate`, `--base_model`).

### Step 3: Eval

Run a side-by-side evaluation to compare the base model and the fine-tuned adapter using sample configurations.

```bash
python scripts/evaluator.py --adapter_path qwen-calculus-finetuned --config_path eval_configs.json
```
Modify `eval_configs.json` in the root directory to add new test problems.

### Step 4: UI Test

Launch the interactive web interface:

```bash
python app.py
```

**Features:**
- Professional Chat Interface with Dark Mode (Monochrome theme).
- Hot-swap adapters: Choose between different checkpoints or unload the adapter directly from the UI without restarting.
- Adjust Generation Settings: Temperature and Top-P sliders.
- Toggle and edit the System Prompt.

Open `http://127.0.0.1:7860` in your browser.

## Optional: Docker

Build and run using the provided Dockerfile:

```bash
docker build -f dockerfile -t calculus-tutor .
docker run --rm -p 7860:7860 calculus-tutor
```

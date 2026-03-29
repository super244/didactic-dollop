# Calculus Tutor Training Platform

A comprehensive training platform for fine-tuning language models on calculus tutoring, supporting multiple hardware configurations from local macOS to large-scale cloud GPU clusters.

## Features

- **Multi-Hardware Support**: macOS MLX/Metal, RTX 5090, RTX Pro 6000, H200, 4xH100, 4xB200, 8xB200
- **Modular Configuration System**: YAML-based configurations for different hardware setups
- **Scalable Dataset Generation**: Automatic dataset preparation with size and difficulty scaling
- **Advanced Training**: LoRA/QLoRA fine-tuning with distributed training support
- **Iterative Fine-Tuning**: Post-training refinement for continuous improvement
- **Web Interface**: Real-time training monitoring, model management, and interactive chat

## Project Structure

```
didactic-dollop/
├── mac/                          # macOS-specific scripts and configs
│   ├── requirements.txt          # macOS dependencies (MLX, etc.)
│   ├── configs/
│   │   └── training_config.yaml  # macOS training configuration
│   ├── scripts/
│   │   ├── config_loader.py      # Configuration management
│   │   ├── prepare_dataset.py    # Dataset generation
│   │   ├── train.py              # MLX training script
│   │   ├── iterate.py            # Iterative fine-tuning
│   │   ├── evaluate.py           # Model evaluation
│   │   └── web_inference.py      # Gradio web interface
│   └── data/                     # Generated datasets
│
├── cloud/                        # Cloud GPU configurations
│   ├── rtx_5090/                 # Consumer GPU (24GB VRAM)
│   │   ├── requirements.txt
│   │   ├── configs/training_config.yaml
│   │   └── scripts/...
│   ├── rtx_pro_6000/             # Workstation GPU (48GB VRAM)
│   ├── h200/                     # High-end single GPU (141GB VRAM)
│   ├── 4x_h100/                  # Multi-GPU (4x 80GB)
│   ├── 4x_b200/                  # Multi-GPU with DeepSpeed (4x 192GB)
│   └── 8x_b200/                  # Large-scale cluster (8x 192GB)
│
└── README.md
```

## Quick Start

### macOS Training

```bash
# 1. Install dependencies
cd mac
pip install -r requirements.txt

# 2. Prepare dataset
python scripts/prepare_dataset.py

# 3. Train model
python scripts/train.py

# 4. Iterative fine-tuning (optional)
python scripts/iterate.py --checkpoint outputs/best_model --iterations 3

# 5. Evaluate
python scripts/evaluate.py --model outputs/best_model

# 6. Launch web interface
python scripts/web_inference.py
```

### Cloud GPU Training

```bash
# 1. Navigate to your GPU configuration
cd cloud/rtx_5090  # or h200, 4x_h100, etc.

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare dataset
python scripts/prepare_dataset.py

# 4. Train (single GPU)
python scripts/train.py

# For multi-GPU (e.g., 4x_h100)
torchrun --nproc_per_node=4 scripts/train.py

# For large-scale with DeepSpeed (e.g., 8x_b200)
torchrun --nproc_per_node=8 scripts/train.py

# 5. Evaluate
python scripts/evaluate.py --model outputs/final_model

# 6. Launch web interface
python scripts/web_inference.py
```

## Hardware Configurations

| Configuration | GPU Memory | Model Size | Dataset Size | Features |
|---------------|------------|------------|--------------|----------|
| macOS MLX | 64GB unified | 0.5B | 5K | Metal optimization |
| RTX 5090 | 24GB | 1.5B | 20K | 4-bit QLoRA |
| RTX Pro 6000 | 48GB | 3B | 35K | 4-bit QLoRA |
| H200 | 141GB | 7B | 50K | Large model support |
| 4x H100 | 4x 80GB | 14B | 100K | Distributed training |
| 4x B200 | 4x 192GB | 32B | 200K | DeepSpeed integration |
| 8x B200 | 8x 192GB | 72B | 500K | Large-scale cluster |

## Configuration Details

Each GPU configuration includes:

- **requirements.txt**: Hardware-specific dependencies
- **configs/training_config.yaml**: Training parameters, model settings, dataset configuration
- **scripts/**: Complete set of scripts for the configuration
  - `config_loader.py`: Load and manage configuration
  - `prepare_dataset.py`: Generate calculus problems
  - `train.py`: Training script with LoRA/QLoRA
  - `iterate.py`: Iterative fine-tuning
  - `evaluate.py`: Benchmark evaluation
  - `web_inference.py`: Gradio web interface

## Dataset Generation

The dataset generator creates calculus problems across five types:

- **Derivatives**: Basic rules, chain rule, implicit differentiation
- **Integrals**: Basic, substitution, integration by parts
- **Limits**: Direct evaluation, L'Hopital's rule, series
- **Series**: Maclaurin, Taylor, convergence
- **Applications**: Optimization, related rates, volume

Difficulty progression scales with hardware capability.

## Training Features

### LoRA/QLoRA Fine-Tuning
- Memory-efficient training with 4-bit quantization
- Configurable LoRA rank and alpha parameters
- Target modules for comprehensive adaptation

### Distributed Training
- Automatic multi-GPU setup with torchrun
- DeepSpeed integration for large models
- Gradient accumulation for memory optimization

### Iterative Fine-Tuning
- Progressive difficulty scaling
- Multi-step and word problem generation
- Automatic best model selection

## Web Interface

Launch the Gradio web interface for:

- Interactive chat with trained models
- Model and adapter selection
- Generation parameter tuning
- Real-time responses

## Environment Variables

```bash
# HuggingFace authentication
export HF_TOKEN=your_token_here

# Base model (optional, defaults in config)
export BASE_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
```

## Requirements

### macOS
- Python 3.8+
- MLX (`pip install mlx`)
- Apple Silicon Mac

### Cloud GPU
- Python 3.8+
- CUDA-compatible GPU
- PyTorch with CUDA support
- bitsandbytes for quantization
- DeepSpeed (for multi-GPU large models)

## License

[Your License Here]

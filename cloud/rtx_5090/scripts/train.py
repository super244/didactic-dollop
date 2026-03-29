#!/usr/bin/env python3
"""
RTX 5090 Training Script
Consumer GPU with 24GB VRAM.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config_loader import ConfigLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RTX5090Trainer:
    """Trainer for RTX 5090 configuration."""
    
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.get_full_config()
        self.training_config = self.config_loader.get_training_config()
        self.dataset_config = self.config_loader.get_dataset_config()
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. This script requires GPU support.")
        
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with QLoRA quantization."""
        model_config = self.config['model']
        
        model_name = model_config['base_model']
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=os.environ.get('HF_TOKEN')
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup quantization
        bnb_config = None
        if model_config.get('use_qlora', False) and model_config.get('quantization') == '4bit':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit QLoRA quantization")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=os.environ.get('HF_TOKEN'),
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        # Prepare for k-bit training
        if model_config.get('use_qlora', False):
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA
        self.setup_lora()
        
        logger.info("Model and tokenizer loaded successfully")
    
    def setup_lora(self):
        """Setup LoRA configuration."""
        training_params = self.training_config
        
        lora_config = LoraConfig(
            r=training_params['lora_r'],
            lora_alpha=training_params['lora_alpha'],
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=training_params['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def load_dataset(self):
        """Load and prepare dataset."""
        dataset_path = Path(__file__).parent.parent / "data" / "calculus_problems.jsonl"
        
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            logger.info("Please run dataset preparation first:")
            logger.info("python cloud/rtx_5090/scripts/prepare_dataset.py")
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        logger.info(f"Loading dataset from {dataset_path}")
        
        # Load dataset
        dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        
        # Tokenization function
        max_length = self.training_config['max_length']
        
        def tokenize_function(examples):
            formatted = [
                f"Problem: {problem}\nSolution: {solution}"
                for problem, solution in zip(examples["problem"], examples["solution"])
            ]
            return self.tokenizer(
                formatted,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
        
        # Tokenize dataset
        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        # Split dataset
        split_ratio = self.dataset_config['split_ratio']
        train_size = int(len(tokenized_dataset) * split_ratio)
        
        train_dataset = tokenized_dataset.select(range(train_size))
        val_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
        
        logger.info(f"Dataset split: {len(train_dataset)} training, {len(val_dataset)} validation examples")
        
        return train_dataset, val_dataset
    
    def setup_training_args(self):
        """Setup training arguments."""
        output_config = self.config['output']
        
        # Setup output directory
        output_dir = Path(output_config['base_dir'])
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.training_config['num_epochs'],
            per_device_train_batch_size=self.training_config['batch_size'],
            per_device_eval_batch_size=self.training_config['batch_size'],
            gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
            learning_rate=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay'],
            logging_steps=output_config['logging_steps'],
            save_strategy=output_config['save_strategy'],
            evaluation_strategy="epoch",
            warmup_ratio=self.training_config['warmup_ratio'],
            lr_scheduler_type="cosine",
            fp16=True,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to="none",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        return training_args
    
    def train(self):
        """Main training function."""
        logger.info("Starting RTX 5090 training")
        
        # Load components
        self.load_model_and_tokenizer()
        train_dataset, val_dataset = self.load_dataset()
        
        # Setup training
        training_args = self.setup_training_args()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Setup trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        
        # Start training
        logger.info("Training started...")
        self.trainer.train()
        
        # Save final model
        output_dir = Path(self.config['output']['base_dir']) / "final_model"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        
        # Save configuration
        with open(output_dir / "training_config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Training completed! Model saved to {output_dir}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train model on RTX 5090")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = RTX5090Trainer()
    
    # Start training
    try:
        trainer.train()
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

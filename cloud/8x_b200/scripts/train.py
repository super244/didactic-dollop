#!/usr/bin/env python3
"""Training Script for 8x_b200"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments, EarlyStoppingCallback
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class 8xb200Trainer:
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.get_full_config()
        self.training_config = self.config_loader.get_training_config()
        self.dataset_config = self.config_loader.get_dataset_config()
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        logger.info(f"CUDA: {torch.cuda.get_device_name(0)}")
        
        # Setup distributed training
        if hardware.get('multi_gpu', False) and hardware.get('num_gpus', 1) > 1:
            if 'LOCAL_RANK' in os.environ:
                self.local_rank = int(os.environ['LOCAL_RANK'])
                self.world_size = int(os.environ['WORLD_SIZE'])
                dist.init_process_group(backend='nccl')
                torch.cuda.set_device(self.local_rank)
                self.is_distributed = True
                logger.info(f"Distributed training: rank {self.local_rank}, world size {self.world_size}")
            else:
                self.is_distributed = False
                self.local_rank = 0
        else:
            self.is_distributed = False
            self.local_rank = 0

        
        self.model = None
        self.tokenizer = None
    
    def load_model_and_tokenizer(self):
        model_name = self.config['model']['base_model']
        logger.info(f"Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ.get('HF_TOKEN'))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        device_map = "auto" if self.config['hardware'].get('num_gpus', 1) > 1 else None
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, token=os.environ.get('HF_TOKEN'),
            quantization_config=bnb_config, device_map=device_map, torch_dtype=torch.float16
        )
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=self.training_config['lora_r'],
            lora_alpha=self.training_config['lora_alpha'],
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=self.training_config['lora_dropout'],
            bias="none", task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def load_dataset(self):
        dataset_path = Path(__file__).parent.parent / "data" / "calculus_problems.jsonl"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        
        def tokenize(examples):
            formatted = [f"Problem: {p}\nSolution: {s}" for p, s in zip(examples["problem"], examples["solution"])]
            return self.tokenizer(formatted, truncation=True, padding="max_length", max_length=self.training_config['max_length'])
        
        tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
        
        split = int(len(tokenized) * self.dataset_config['split_ratio'])
        return tokenized.select(range(split)), tokenized.select(range(split, len(tokenized)))
    
    def train(self):
        self.load_model_and_tokenizer()
        train_ds, val_ds = self.load_dataset()
        
        output_dir = Path(self.config['output']['base_dir'])
        
    def setup_deepspeed(self):
        if not self.training_config.get('deepspeed', False):
            return None
        
        ds_config = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "zero_optimization": {"stage": 2},
            "fp16": {"enabled": True}
        }
        
        ds_path = Path(self.config['output']['base_dir']) / "deepspeed_config.json"
        ds_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ds_path, 'w') as f:
            json.dump(ds_config, f)
        return str(ds_path)

        
        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.training_config['num_epochs'],
            per_device_train_batch_size=self.training_config['batch_size'],
            gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
            learning_rate=float(self.training_config['learning_rate']),
            fp16=True,
            logging_steps=100,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if hasattr(self, 'is_distributed') and self.is_distributed else None,
            deepspeed=self.setup_deepspeed() if hasattr(self, 'setup_deepspeed') else None,
        )
        
        trainer = Trainer(
            model=self.model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        trainer.train()
        
        final_dir = output_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))
        
        logger.info(f"Training completed! Model saved to {final_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train on 8x_b200")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    trainer = 8xb200Trainer()
    try:
        trainer.train()
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

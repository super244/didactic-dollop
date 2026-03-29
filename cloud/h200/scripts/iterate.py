#!/usr/bin/env python3
"""Iterative Fine-Tuning Script for h200"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
import random

import torch
from datasets import load_dataset
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FineTuner:
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.get_full_config()
        self.training_config = self.config_loader.get_training_config()
        self.fine_tuning_config = self.config_loader.get_fine_tuning_config()
        self.model = None
        self.tokenizer = None
    
    def load_base_model(self, checkpoint_path: str):
        logger.info(f"Loading from {checkpoint_path}")
        checkpoint = Path(checkpoint_path)
        
        self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        base = AutoModelForCausalLM.from_pretrained(
            self.config['model']['base_model'], token=os.environ.get('HF_TOKEN'),
            quantization_config=bnb, torch_dtype=torch.float16
        )
        base = prepare_model_for_kbit_training(base)
        
        if (checkpoint / "adapter_config.json").exists():
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base, str(checkpoint))
        else:
            self.model = base
    
    def generate_data(self, iteration: int):
        dataset_path = Path(__file__).parent.parent / "data" / "calculus_problems.jsonl"
        problems = []
        with open(dataset_path) as f:
            for line in f: problems.append(json.loads(line))
        
        if iteration >= 1: problems.extend([{"problem": f"Multi-step {i}", "solution": "Solution"} for i in range(200)])
        if iteration >= 2: problems.extend([{"problem": f"Word problem {i}", "solution": "Solution"} for i in range(200)])
        
        random.shuffle(problems)
        return problems
    
    def prepare_dataset(self, problems):
        path = Path(__file__).parent.parent / "data" / "ft_temp.jsonl"
        with open(path, 'w') as f:
            for p in problems: f.write(json.dumps(p) + '\n')
        
        ds = load_dataset("json", data_files=str(path), split="train")
        def tokenize(ex):
            fmt = [f"Problem: {p}\nSolution: {s}" for p, s in zip(ex["problem"], ex["solution"])]
            return self.tokenizer(fmt, truncation=True, padding="max_length", max_length=self.training_config['max_length'])
        return ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    
    def fine_tune_iteration(self, iteration: int, dataset):
        lr = float(self.training_config['learning_rate']) * (self.fine_tuning_config['learning_rate_decay'] ** iteration)
        output_dir = Path(self.config['output']['base_dir']) / f"iteration_{iteration}"
        
        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.fine_tuning_config['epochs_per_iteration'],
            per_device_train_batch_size=self.training_config['batch_size'],
            gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
            learning_rate=lr, fp16=True, logging_steps=50, save_strategy="epoch"
        )
        
        trainer = Trainer(model=self.model, args=args, train_dataset=dataset)
        trainer.train()
        
        iter_dir = Path(self.config['output']['base_dir']) / f"fine_tuned_iteration_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(iter_dir))
        self.tokenizer.save_pretrained(str(iter_dir))
        
        logger.info(f"Iteration {iteration} saved to {iter_dir}")
    
    def fine_tune(self, num_iterations: int = None):
        if num_iterations is None:
            num_iterations = self.fine_tuning_config['num_iterations']
        
        for i in range(1, num_iterations + 1):
            problems = self.generate_data(i)
            dataset = self.prepare_dataset(problems)
            self.fine_tune_iteration(i, dataset)
        
        logger.info("Fine-tuning completed!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on h200")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args()
    
    tuner = FineTuner()
    try:
        tuner.load_base_model(args.checkpoint)
        tuner.fine_tune(args.iterations)
        return 0
    except Exception as e:
        logger.error(f"Failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

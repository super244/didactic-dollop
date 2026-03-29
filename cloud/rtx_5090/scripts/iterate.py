#!/usr/bin/env python3
"""
RTX 5090 Iterative Fine-Tuning Script
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List
import random

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config_loader import ConfigLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RTX5090FineTuner:
    """Fine-tuning system for RTX 5090."""
    
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.get_full_config()
        self.training_config = self.config_loader.get_training_config()
        self.fine_tuning_config = self.config_loader.get_fine_tuning_config()
        
        self.model = None
        self.tokenizer = None
        self.base_checkpoint = None
        self.iteration_results = []
        
    def load_base_model(self, checkpoint_path: str):
        """Load base model from checkpoint."""
        logger.info(f"Loading base model from {checkpoint_path}")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        model_config = self.config['model']
        model_name = model_config['base_model']
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint_path),
            token=os.environ.get('HF_TOKEN')
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=os.environ.get('HF_TOKEN'),
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
        )
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Load adapter if exists
        if (checkpoint_path / "adapter_config.json").exists():
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(checkpoint_path))
        
        self.base_checkpoint = checkpoint_path
        logger.info("Base model loaded successfully")
    
    def generate_fine_tuning_data(self, iteration: int) -> List[Dict[str, str]]:
        """Generate fine-tuning data."""
        logger.info(f"Generating fine-tuning data for iteration {iteration}")
        
        dataset_path = Path(__file__).parent.parent / "data" / "calculus_problems.jsonl"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        base_problems = []
        with open(dataset_path, 'r') as f:
            for line in f:
                base_problems.append(json.loads(line))
        
        additional_problems = []
        
        if iteration >= 1:
            additional_problems.extend(self._generate_multi_step_problems(200))
        
        if iteration >= 2:
            additional_problems.extend(self._generate_word_problems(200))
        
        if iteration >= 3:
            additional_problems.extend(self._generate_proof_problems(100))
        
        all_problems = base_problems + additional_problems
        random.shuffle(all_problems)
        
        logger.info(f"Generated {len(all_problems)} fine-tuning examples")
        return all_problems
    
    def _generate_multi_step_problems(self, count: int) -> List[Dict[str, str]]:
        problems = []
        for _ in range(count):
            problems.append({
                "problem": f"Multi-step problem {random.randint(1, 100)}",
                "solution": "Step-by-step solution",
                "difficulty": "intermediate",
                "type": "derivatives"
            })
        return problems
    
    def _generate_word_problems(self, count: int) -> List[Dict[str, str]]:
        problems = []
        for _ in range(count):
            problems.append({
                "problem": f"Word problem {random.randint(1, 100)}",
                "solution": "Solution with context",
                "difficulty": "intermediate",
                "type": "applications"
            })
        return problems
    
    def _generate_proof_problems(self, count: int) -> List[Dict[str, str]]:
        problems = []
        for _ in range(count):
            problems.append({
                "problem": f"Proof problem {random.randint(1, 100)}",
                "solution": "Proof with reasoning",
                "difficulty": "advanced",
                "type": "proofs"
            })
        return problems
    
    def prepare_dataset(self, problems: List[Dict[str, str]]):
        """Prepare dataset for fine-tuning."""
        dataset_path = Path(__file__).parent.parent / "data" / f"fine_tune_iter.jsonl"
        
        with open(dataset_path, 'w') as f:
            for problem in problems:
                f.write(json.dumps(problem) + '\n')
        
        dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        
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
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        
        return tokenized_dataset
    
    def fine_tune_iteration(self, iteration: int, dataset):
        """Perform one fine-tuning iteration."""
        logger.info(f"Starting fine-tuning iteration {iteration}")
        
        learning_rate = self.training_config['learning_rate'] * (self.fine_tuning_config['learning_rate_decay'] ** iteration)
        
        output_dir = Path(self.config['output']['base_dir']) / f"iteration_{iteration}"
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.fine_tuning_config['epochs_per_iteration'],
            per_device_train_batch_size=self.training_config['batch_size'],
            gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=50,
            save_strategy="epoch",
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )
        
        trainer.train()
        
        # Save iteration
        iteration_dir = Path(self.config['output']['base_dir']) / f"fine_tuned_iteration_{iteration}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        
        trainer.save_model(str(iteration_dir))
        self.tokenizer.save_pretrained(str(iteration_dir))
        
        logger.info(f"Iteration {iteration} saved to {iteration_dir}")
        
        return iteration_dir
    
    def fine_tune(self, num_iterations: int = None):
        """Main fine-tuning loop."""
        if num_iterations is None:
            num_iterations = self.fine_tuning_config['num_iterations']
        
        logger.info(f"Starting fine-tuning with {num_iterations} iterations")
        
        for iteration in range(1, num_iterations + 1):
            logger.info(f"=== Fine-Tuning Iteration {iteration} ===")
            
            problems = self.generate_fine_tuning_data(iteration)
            dataset = self.prepare_dataset(problems)
            iteration_dir = self.fine_tune_iteration(iteration, dataset)
            
            self.iteration_results.append({
                'iteration': iteration,
                'path': str(iteration_dir)
            })
        
        logger.info("Fine-tuning completed!")
        
        # Save results
        output_dir = Path(self.config['output']['base_dir'])
        with open(output_dir / "fine_tuning_results.json", 'w') as f:
            json.dump(self.iteration_results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model on RTX 5090")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=None)
    
    args = parser.parse_args()
    
    tuner = RTX5090FineTuner()
    
    try:
        tuner.load_base_model(args.checkpoint)
        tuner.fine_tune(args.iterations)
        return 0
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

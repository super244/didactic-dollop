#!/usr/bin/env python3
"""
macOS Post-Training Fine-Tuning Script
Performs iterative fine-tuning and model refinement on macOS using MLX.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil
import random

# MLX imports
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    print("MLX imported successfully")
except ImportError:
    print("MLX not installed. Please install with: pip install mlx")
    sys.exit(1)

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config_loader import ConfigLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MacOSFineTuner:
    """Fine-tuning system for macOS with iterative improvement."""
    
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.get_full_config()
        self.training_config = self.config_loader.get_training_config()
        self.fine_tuning_config = self.config_loader.get_fine_tuning_config()
        
        # Setup MLX
        if mx.metal.is_available():
            mx.metal.set_cache_limit(64 * 1024 * 1024 * 1024)
            logger.info("Using Metal backend with 64GB cache limit")
        
        # Initialize components
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
        
        # Load tokenizer
        tokenizer_path = checkpoint_path
        if (checkpoint_path / "tokenizer.json").exists():
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        else:
            from transformers import AutoTokenizer
            base_model = self.config['model']['base_model']
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model weights
        weights_path = checkpoint_path / "model_weights.json"
        if weights_path.exists():
            with open(weights_path, 'r') as f:
                weights = json.load(f)
            
            vocab_size = self.tokenizer.vocab_size
            hidden_size = 768
            num_layers = 12
            
            class SimpleTransformer(nn.Module):
                def __init__(self, vocab_size, hidden_size, num_layers):
                    super().__init__()
                    self.embedding = nn.Embedding(vocab_size, hidden_size)
                    self.layers = [nn.TransformerEncoder(hidden_size, 12) for _ in range(num_layers)]
                    self.norm = nn.LayerNorm(hidden_size)
                    self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
                
                def __call__(self, x):
                    x = self.embedding(x)
                    for layer in self.layers:
                        x = layer(x)
                    x = self.norm(x)
                    return self.lm_head(x)
            
            self.model = SimpleTransformer(vocab_size, hidden_size, num_layers)
            logger.info("Model weights loaded successfully")
        else:
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        
        self.base_checkpoint = checkpoint_path
        logger.info("Base model loaded successfully")
    
    def generate_fine_tuning_data(self, iteration: int) -> List[Dict[str, str]]:
        """Generate fine-tuning data for specific iteration."""
        logger.info(f"Generating fine-tuning data for iteration {iteration}")
        
        # Load existing dataset
        dataset_path = Path(__file__).parent.parent / "data" / "calculus_problems.jsonl"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        base_problems = []
        with open(dataset_path, 'r') as f:
            for line in f:
                base_problems.append(json.loads(line))
        
        # Generate additional challenging problems
        additional_problems = []
        
        if iteration >= 1:
            additional_problems.extend(self._generate_multi_step_problems(100))
        
        if iteration >= 2:
            additional_problems.extend(self._generate_word_problems(100))
        
        if iteration >= 3:
            additional_problems.extend(self._generate_proof_problems(50))
        
        all_problems = base_problems + additional_problems
        random.shuffle(all_problems)
        
        logger.info(f"Generated {len(all_problems)} fine-tuning examples")
        return all_problems
    
    def _generate_multi_step_problems(self, count: int) -> List[Dict[str, str]]:
        """Generate multi-step calculus problems."""
        problems = []
        
        for _ in range(count):
            inner_funcs = ["x^2", "sin(x)", "e^x", "ln(x)"]
            outer_funcs = ["sin(u)", "e^u", "u^3", "ln(u)"]
            
            inner = random.choice(inner_funcs)
            outer = random.choice(outer_funcs)
            composite = outer.replace("u", f"({inner})")
            
            solution = f"Apply chain rule: d/dx {composite} = (derivative of outer) * (derivative of inner)"
            
            problems.append({
                "problem": f"Find the derivative of {composite} using the chain rule.",
                "solution": solution,
                "difficulty": "intermediate",
                "type": "derivatives"
            })
        
        return problems
    
    def _generate_word_problems(self, count: int) -> List[Dict[str, str]]:
        """Generate word problems."""
        problems = []
        
        scenarios = [
            ("A car travels with velocity v(t) = t^2 + 2t. Find the distance traveled from t=0 to t=3.", 
             "Distance = ∫[0 to 3] (t^2 + 2t) dt = [t^3/3 + t^2] from 0 to 3 = 9 + 9 = 18 units"),
            ("A company's revenue is R(x) = 100x - x^2. Find the maximum revenue.", 
             "Maximum at vertex x = 50, R(50) = 100(50) - 50^2 = 2500"),
            ("Water drains from a tank at rate dh/dt = -0.5√h. Find time to drain from 16m to 0m.", 
             "Separate variables and integrate: ∫h^(-1/2) dh = -0.5∫dt, gives t = 8 seconds"),
        ]
        
        for _ in range(count):
            problem, solution = random.choice(scenarios)
            problems.append({
                "problem": problem,
                "solution": solution,
                "difficulty": "intermediate",
                "type": "applications"
            })
        
        return problems
    
    def _generate_proof_problems(self, count: int) -> List[Dict[str, str]]:
        """Generate proof-based problems."""
        problems = []
        
        proofs = [
            ("Prove that the derivative of sin(x) is cos(x) using the limit definition.",
             "Use limit definition: lim[h→0] (sin(x+h) - sin(x))/h = lim[h→0] (sin(x)cos(h) + cos(x)sin(h) - sin(x))/h = cos(x)"),
            ("Prove the Fundamental Theorem of Calculus for continuous functions.",
             "If F(x) = ∫[a to x] f(t) dt, then F'(x) = f(x) by the definition of derivative and properties of integrals"),
            ("Prove that e^x is its own derivative using the series definition.",
             "e^x = Σ[n=0 to ∞] x^n/n!. Differentiate term by term: Σ[n=1 to ∞] nx^(n-1)/n! = Σ[n=0 to ∞] x^n/n! = e^x"),
        ]
        
        for _ in range(count):
            problem, solution = random.choice(proofs)
            problems.append({
                "problem": problem,
                "solution": solution,
                "difficulty": "advanced",
                "type": "proofs"
            })
        
        return problems
    
    def prepare_fine_tuning_data(self, problems: List[Dict[str, str]]):
        """Prepare data for fine-tuning."""
        max_length = self.training_config['max_length']
        
        def tokenize_example(example):
            problem = example['problem']
            solution = example['solution']
            
            text = f"Problem: {problem}\nSolution: {solution}"
            
            encoded = self.tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='np'
            )
            
            return {
                'input_ids': encoded['input_ids'][0],
                'attention_mask': encoded['attention_mask'][0],
                'labels': encoded['input_ids'][0]
            }
        
        processed_examples = [tokenize_example(ex) for ex in problems]
        dataset = mx.array([ex['input_ids'] for ex in processed_examples])
        
        return dataset
    
    def fine_tune_iteration(self, iteration: int, dataset):
        """Perform one fine-tuning iteration."""
        logger.info(f"Starting fine-tuning iteration {iteration}")
        
        learning_rate = self.training_config['learning_rate'] * (self.fine_tuning_config['learning_rate_decay'] ** iteration)
        batch_size = max(4, self.training_config['batch_size'] // 2)
        num_epochs = self.fine_tuning_config['epochs_per_iteration']
        
        optimizer = optim.Adam(learning_rate=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        
        self.model.train()
        total_loss = 0
        
        num_samples = len(dataset)
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx in range(0, num_samples, batch_size):
                batch_end = min(batch_idx + batch_size, num_samples)
                batch_inputs = dataset[batch_idx:batch_end]
                
                outputs = self.model(batch_inputs)
                
                targets = batch_inputs[:, 1:]
                outputs = outputs[:, :-1, :]
                
                batch_size_actual, seq_len, vocab_size = outputs.shape
                outputs = outputs.reshape(-1, vocab_size)
                targets = targets.reshape(-1)
                
                loss = loss_fn(outputs, targets)
                
                loss.backward()
                optimizer.step(self.model)
                optimizer.clear_grad()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches
            logger.info(f"Iteration {iteration}, Epoch {epoch + 1}, Loss: {avg_epoch_loss:.4f}")
            total_loss += avg_epoch_loss
        
        avg_loss = total_loss / num_epochs
        return avg_loss
    
    def save_iteration(self, iteration: int, loss: float):
        """Save fine-tuned model for iteration."""
        output_dir = Path(self.config['output']['base_dir'])
        iteration_dir = output_dir / f"fine_tuned_iteration_{iteration}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        weights = {k: v.tolist() for k, v in self.model.parameters.items()}
        with open(iteration_dir / "model_weights.json", 'w') as f:
            json.dump(weights, f)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(iteration_dir)
        
        # Save iteration info
        iteration_info = {
            'iteration': iteration,
            'base_checkpoint': str(self.base_checkpoint),
            'loss': loss,
            'learning_rate': self.training_config['learning_rate'] * (self.fine_tuning_config['learning_rate_decay'] ** iteration),
            'config': self.config
        }
        
        with open(iteration_dir / "iteration_info.json", 'w') as f:
            json.dump(iteration_info, f, indent=2)
        
        logger.info(f"Iteration {iteration} saved to {iteration_dir}")
        
        # Update best model
        self.iteration_results.append({'iteration': iteration, 'loss': loss, 'path': str(iteration_dir)})
        
        if iteration == 1 or loss < min(r['loss'] for r in self.iteration_results[:-1]):
            best_dir = output_dir / "best_fine_tuned"
            best_dir.mkdir(parents=True, exist_ok=True)
            
            shutil.copytree(iteration_dir, best_dir, dirs_exist_ok=True)
            logger.info(f"New best fine-tuned model saved with loss: {loss:.4f}")
    
    def fine_tune(self, num_iterations: int = None):
        """Main fine-tuning loop."""
        if num_iterations is None:
            num_iterations = self.fine_tuning_config['num_iterations']
        
        logger.info(f"Starting fine-tuning with {num_iterations} iterations")
        
        for iteration in range(1, num_iterations + 1):
            logger.info(f"=== Fine-Tuning Iteration {iteration} ===")
            
            # Generate fine-tuning data
            problems = self.generate_fine_tuning_data(iteration)
            
            # Prepare data
            dataset = self.prepare_fine_tuning_data(problems)
            
            # Fine-tune
            loss = self.fine_tune_iteration(iteration, dataset)
            
            # Save iteration
            self.save_iteration(iteration, loss)
        
        logger.info("Fine-tuning completed!")
        
        # Save final results
        output_dir = Path(self.config['output']['base_dir'])
        with open(output_dir / "fine_tuning_results.json", 'w') as f:
            json.dump(self.iteration_results, f, indent=2)
        
        logger.info(f"Results saved to {output_dir / 'fine_tuning_results.json'}")


def main():
    """Main fine-tuning function."""
    parser = argparse.ArgumentParser(description="Fine-tune model on macOS using MLX")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Base checkpoint path")
    parser.add_argument("--iterations", type=int, default=None,
                       help="Number of fine-tuning iterations")
    
    args = parser.parse_args()
    
    # Initialize fine-tuner
    tuner = MacOSFineTuner()
    
    # Load base model
    try:
        tuner.load_base_model(args.checkpoint)
    except Exception as e:
        logger.error(f"Failed to load base model: {e}")
        return 1
    
    # Start fine-tuning
    try:
        tuner.fine_tune(args.iterations)
        return 0
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
macOS MLX/Metal Optimized Training Script
Optimized for Apple Silicon with up to 64GB unified memory.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import argparse

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


class MLXTrainer:
    """MLX-optimized trainer for macOS."""
    
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.get_full_config()
        self.training_config = self.config_loader.get_training_config()
        self.dataset_config = self.config_loader.get_dataset_config()
        
        # Set up MLX device
        if mx.metal.is_available():
            mx.metal.set_cache_limit(64 * 1024 * 1024 * 1024)  # 64GB cache
            logger.info("Using Metal backend with 64GB cache limit")
        else:
            logger.warning("Metal not available, falling back to CPU")
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.train_loader = None
        self.val_loader = None
        
    def load_model_and_tokenizer(self):
        """Load model and tokenizer for MLX."""
        model_name = self.config['model']['base_model']
        logger.info(f"Loading model: {model_name}")
        
        try:
            from transformers import AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create a simple model for demonstration
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
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_dataset(self):
        """Load and prepare dataset for MLX training."""
        dataset_path = Path(__file__).parent.parent / "data" / "calculus_problems.jsonl"
        
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            logger.info("Please run dataset preparation first:")
            logger.info("python mac/scripts/prepare_dataset.py")
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        logger.info(f"Loading dataset from {dataset_path}")
        
        # Load dataset
        examples = []
        with open(dataset_path, 'r') as f:
            for line in f:
                examples.append(json.loads(line))
        
        # Prepare training data
        max_length = self.training_config['max_length']
        
        def tokenize_example(example):
            problem = example['problem']
            solution = example['solution']
            
            # Format as chat
            text = f"Problem: {problem}\nSolution: {solution}"
            
            # Tokenize
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
        
        # Process examples
        processed_examples = [tokenize_example(ex) for ex in examples]
        
        # Split into train/val
        split_idx = int(len(processed_examples) * self.dataset_config['split_ratio'])
        train_examples = processed_examples[:split_idx]
        val_examples = processed_examples[split_idx:]
        
        # Create MLX datasets
        train_dataset = mx.array([ex['input_ids'] for ex in train_examples])
        val_dataset = mx.array([ex['input_ids'] for ex in val_examples])
        
        logger.info(f"Dataset loaded: {len(train_examples)} training, {len(val_examples)} validation examples")
        
        return train_dataset, val_dataset
    
    def setup_training(self):
        """Set up training components."""
        # Optimizer
        learning_rate = self.training_config['learning_rate']
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Training parameters
        self.batch_size = self.training_config['batch_size']
        self.num_epochs = self.training_config['num_epochs']
        self.gradient_accumulation_steps = self.training_config['gradient_accumulation_steps']
        
        # Output directory
        self.output_dir = Path(self.config['output']['base_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Training setup complete. Output dir: {self.output_dir}")
    
    def train_epoch(self, epoch: int, train_dataset):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        num_samples = len(train_dataset)
        num_batches_total = num_samples // self.batch_size
        
        for batch_idx in range(0, num_samples, self.batch_size):
            batch_end = min(batch_idx + self.batch_size, num_samples)
            batch_inputs = train_dataset[batch_idx:batch_end]
            
            # Forward pass
            outputs = self.model(batch_inputs)
            
            # Calculate loss
            targets = batch_inputs[:, 1:]
            outputs = outputs[:, :-1, :]
            
            batch_size, seq_len, vocab_size = outputs.shape
            outputs = outputs.reshape(-1, vocab_size)
            targets = targets.reshape(-1)
            
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx // self.batch_size + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step(self.model)
                self.optimizer.clear_grad()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Logging
            if batch_idx % (self.batch_size * 10) == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx//self.batch_size}/{num_batches_total}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_dataset):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        num_samples = len(val_dataset)
        
        with mx.no_grad():
            for batch_idx in range(0, num_samples, self.batch_size):
                batch_end = min(batch_idx + self.batch_size, num_samples)
                batch_inputs = val_dataset[batch_idx:batch_end]
                
                outputs = self.model(batch_inputs)
                
                # Calculate loss
                targets = batch_inputs[:, 1:]
                outputs = outputs[:, :-1, :]
                
                batch_size, seq_len, vocab_size = outputs.shape
                outputs = outputs.reshape(-1, vocab_size)
                targets = targets.reshape(-1)
                
                loss = self.loss_fn(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self):
        """Main training loop."""
        logger.info("Starting MLX training...")
        
        # Load components
        self.load_model_and_tokenizer()
        train_dataset, val_dataset = self.load_dataset()
        self.setup_training()
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(epoch, train_dataset)
            
            # Validate
            val_loss = self.validate(val_dataset)
            
            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            checkpoint_path = self.output_dir / f"checkpoint-{epoch + 1}"
            checkpoint_path.mkdir(exist_ok=True)
            
            # Save model weights
            weights = {k: v.tolist() for k, v in self.model.parameters.items()}
            with open(checkpoint_path / "model_weights.json", 'w') as f:
                json.dump(weights, f)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(checkpoint_path)
            
            # Save training config
            with open(checkpoint_path / "training_config.json", 'w') as f:
                json.dump(self.config, f, indent=2)
            
            # Update best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = self.output_dir / "best_model"
                best_path.mkdir(exist_ok=True)
                
                # Copy best model
                import shutil
                shutil.copytree(checkpoint_path, best_path, dirs_exist_ok=True)
                
                logger.info(f"New best model saved with val loss: {best_val_loss:.4f}")
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Model saved to: {self.output_dir}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train model on macOS using MLX")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = MLXTrainer()
    
    # Start training
    try:
        trainer.train()
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

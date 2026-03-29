#!/usr/bin/env python3
"""
Evaluation Script for macOS
Evaluates trained models on public benchmarks.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import random

# MLX imports
try:
    import mlx.core as mx
    import mlx.nn as nn
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


class ModelEvaluator:
    """Evaluator for macOS-trained models."""
    
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.get_full_config()
        self.evaluation_config = self.config_loader.get_evaluation_config()
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        
    def load_model(self, model_path: str):
        """Load trained model from path."""
        logger.info(f"Loading model from {model_path}")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load tokenizer
        from transformers import AutoTokenizer
        if (model_path / "tokenizer.json").exists():
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        else:
            base_model = self.config['model']['base_model']
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model weights
        weights_path = model_path / "model_weights.json"
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
            logger.info("Model loaded successfully")
        else:
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
    
    def load_benchmark(self, benchmark_path: str = None) -> List[Dict[str, Any]]:
        """Load evaluation benchmark."""
        if benchmark_path is None:
            benchmark_path = self.evaluation_config.get('benchmark_path', 'data/evaluation_benchmark.json')
        
        benchmark_path = Path(__file__).parent.parent / benchmark_path
        
        if not benchmark_path.exists():
            logger.warning(f"Benchmark not found at {benchmark_path}, generating default benchmark")
            return self._generate_default_benchmark()
        
        with open(benchmark_path, 'r') as f:
            benchmark = json.load(f)
        
        logger.info(f"Loaded {len(benchmark)} benchmark problems")
        return benchmark
    
    def _generate_default_benchmark(self) -> List[Dict[str, Any]]:
        """Generate a default evaluation benchmark."""
        benchmark = [
            {
                "problem": "Find the derivative of sin(x) with respect to x.",
                "expected_keywords": ["cos(x)"],
                "type": "derivatives",
                "difficulty": "basic"
            },
            {
                "problem": "Compute the integral of e^x dx.",
                "expected_keywords": ["e^x", "+ C"],
                "type": "integrals",
                "difficulty": "basic"
            },
            {
                "problem": "Evaluate the limit of sin(x)/x as x -> 0.",
                "expected_keywords": ["1"],
                "type": "limits",
                "difficulty": "basic"
            },
            {
                "problem": "Find the derivative of sin(2x) using chain rule.",
                "expected_keywords": ["2cos(2x)", "chain rule"],
                "type": "derivatives",
                "difficulty": "intermediate"
            },
            {
                "problem": "Compute the integral of x*sin(x) dx using integration by parts.",
                "expected_keywords": ["-x*cos(x)", "sin(x)", "+ C"],
                "type": "integrals",
                "difficulty": "intermediate"
            },
            {
                "problem": "Find the derivative for: x^2 + y^2 = 25",
                "expected_keywords": ["dy/dx", "-x/y"],
                "type": "derivatives",
                "difficulty": "advanced"
            },
            {
                "problem": "Write the Maclaurin series for e^x up to the x^3 term.",
                "expected_keywords": ["1", "x", "x^2/2", "x^3/6"],
                "type": "series",
                "difficulty": "basic"
            },
            {
                "problem": "A ball is thrown upward with velocity 20 m/s. Find its maximum height.",
                "expected_keywords": ["20.4", "meters"],
                "type": "applications",
                "difficulty": "basic"
            },
        ]
        
        logger.info(f"Generated default benchmark with {len(benchmark)} problems")
        return benchmark
    
    def generate_response(self, problem: str, max_tokens: int = 200) -> str:
        """Generate response for a problem."""
        # Format input
        text = f"Problem: {problem}\nSolution:"
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='np',
            max_length=512,
            truncation=True
        )
        
        input_ids = mx.array(inputs['input_ids'])
        
        # Generate (simplified - in practice you'd use proper generation)
        self.model.eval()
        
        with mx.no_grad():
            outputs = self.model(input_ids)
            
            # Get predicted tokens (simplified)
            predicted_ids = mx.argmax(outputs, axis=-1)
            
            # Decode
            response = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        
        return response
    
    def evaluate_problem(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single problem."""
        problem = problem_data['problem']
        expected_keywords = problem_data.get('expected_keywords', [])
        
        # Generate response
        response = self.generate_response(problem)
        
        # Evaluate
        keywords_found = []
        for keyword in expected_keywords:
            if keyword.lower() in response.lower():
                keywords_found.append(keyword)
        
        keyword_score = len(keywords_found) / len(expected_keywords) if expected_keywords else 0.0
        
        # Check for step-by-step reasoning
        has_steps = any(word in response.lower() for word in ['step', 'first', 'then', 'next', 'therefore'])
        
        # Check for mathematical notation
        has_math = any(symbol in response for symbol in ['∫', '∂', '√', '∑', '∞', '=', '+', '-'])
        
        return {
            'problem': problem,
            'response': response,
            'expected_keywords': expected_keywords,
            'keywords_found': keywords_found,
            'keyword_score': keyword_score,
            'has_steps': has_steps,
            'has_math': has_math,
            'type': problem_data.get('type', 'unknown'),
            'difficulty': problem_data.get('difficulty', 'unknown')
        }
    
    def evaluate_all(self, benchmark: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate all problems in benchmark."""
        results = []
        
        for problem_data in benchmark:
            result = self.evaluate_problem(problem_data)
            results.append(result)
            
            logger.info(f"Evaluated: {problem_data['type']}/{problem_data['difficulty']} - Score: {result['keyword_score']:.2f}")
        
        # Calculate aggregate metrics
        total_score = sum(r['keyword_score'] for r in results) / len(results)
        
        by_type = {}
        for result in results:
            ptype = result['type']
            if ptype not in by_type:
                by_type[ptype] = []
            by_type[ptype].append(result['keyword_score'])
        
        by_difficulty = {}
        for result in results:
            diff = result['difficulty']
            if diff not in by_difficulty:
                by_difficulty[diff] = []
            by_difficulty[diff].append(result['keyword_score'])
        
        aggregate = {
            'total_problems': len(results),
            'average_score': total_score,
            'by_type': {k: sum(v)/len(v) for k, v in by_type.items()},
            'by_difficulty': {k: sum(v)/len(v) for k, v in by_difficulty.items()},
            'has_steps_ratio': sum(1 for r in results if r['has_steps']) / len(results),
            'has_math_ratio': sum(1 for r in results if r['has_math']) / len(results),
            'detailed_results': results
        }
        
        return aggregate
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained model on benchmarks")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--benchmark", type=str, default=None,
                       help="Path to benchmark file")
    parser.add_argument("--output", type=str, default="outputs/evaluation_results.json",
                       help="Output path for results")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load model
    try:
        evaluator.load_model(args.model)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Load benchmark
    benchmark = evaluator.load_benchmark(args.benchmark)
    
    # Evaluate
    try:
        results = evaluator.evaluate_all(benchmark)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Total Problems: {results['total_problems']}")
        print(f"Average Score: {results['average_score']:.2%}")
        print(f"\nBy Type:")
        for ptype, score in results['by_type'].items():
            print(f"  {ptype}: {score:.2%}")
        print(f"\nBy Difficulty:")
        for diff, score in results['by_difficulty'].items():
            print(f"  {diff}: {score:.2%}")
        print(f"\nHas Steps: {results['has_steps_ratio']:.2%}")
        print(f"Has Math: {results['has_math_ratio']:.2%}")
        print("="*60 + "\n")
        
        # Save results
        evaluator.save_results(results, args.output)
        
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

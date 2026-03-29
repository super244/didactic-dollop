#!/usr/bin/env python3
"""
Evaluation Script for RTX 5090
Evaluates trained models on public benchmarks.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List
import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config_loader import ConfigLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator for RTX 5090 trained models."""
    
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.get_full_config()
        self.evaluation_config = self.config_loader.get_evaluation_config()
        
        self.model = None
        self.tokenizer = None
        
    def load_model(self, model_path: str):
        """Load trained model."""
        logger.info(f"Loading model from {model_path}")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model_config = self.config['model']
        base_model = model_config['base_model']
        
        # Load tokenizer
        if (model_path / "tokenizer.json").exists():
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, token=os.environ.get('HF_TOKEN'))
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            token=os.environ.get('HF_TOKEN'),
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
        )
        
        # Load adapter if exists
        if (model_path / "adapter_config.json").exists():
            self.model = PeftModel.from_pretrained(base, str(model_path))
        else:
            self.model = base
        
        logger.info("Model loaded successfully")
    
    def load_benchmark(self, benchmark_path: str = None) -> List[Dict[str, Any]]:
        """Load evaluation benchmark."""
        if benchmark_path is None:
            benchmark_path = self.evaluation_config.get('benchmark_path', 'data/evaluation_benchmark.json')
        
        benchmark_path = Path(__file__).parent.parent / benchmark_path
        
        if not benchmark_path.exists():
            logger.warning(f"Benchmark not found, generating default")
            return self._generate_default_benchmark()
        
        with open(benchmark_path, 'r') as f:
            benchmark = json.load(f)
        
        logger.info(f"Loaded {len(benchmark)} benchmark problems")
        return benchmark
    
    def _generate_default_benchmark(self) -> List[Dict[str, Any]]:
        """Generate default benchmark."""
        benchmark = [
            {"problem": "Find the derivative of sin(x).", "expected_keywords": ["cos(x)"], "type": "derivatives", "difficulty": "basic"},
            {"problem": "Compute the integral of e^x dx.", "expected_keywords": ["e^x", "+ C"], "type": "integrals", "difficulty": "basic"},
            {"problem": "Evaluate the limit of sin(x)/x as x -> 0.", "expected_keywords": ["1"], "type": "limits", "difficulty": "basic"},
            {"problem": "Find the derivative of sin(2x).", "expected_keywords": ["2cos(2x)"], "type": "derivatives", "difficulty": "intermediate"},
            {"problem": "Compute the integral of x*sin(x) dx.", "expected_keywords": ["-x*cos(x)", "sin(x)"], "type": "integrals", "difficulty": "intermediate"},
            {"problem": "Find dy/dx for x^2 + y^2 = 25.", "expected_keywords": ["dy/dx", "-x/y"], "type": "derivatives", "difficulty": "advanced"},
            {"problem": "Write the Maclaurin series for e^x.", "expected_keywords": ["1", "x", "x^2/2"], "type": "series", "difficulty": "basic"},
            {"problem": "A ball is thrown upward at 20 m/s. Find max height.", "expected_keywords": ["20.4", "meters"], "type": "applications", "difficulty": "basic"},
        ]
        return benchmark
    
    def generate_response(self, problem: str, max_tokens: int = 200) -> str:
        """Generate response."""
        text = f"Problem: {problem}\nSolution:"
        
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    def evaluate_problem(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate single problem."""
        problem = problem_data['problem']
        expected_keywords = problem_data.get('expected_keywords', [])
        
        response = self.generate_response(problem)
        
        keywords_found = [kw for kw in expected_keywords if kw.lower() in response.lower()]
        keyword_score = len(keywords_found) / len(expected_keywords) if expected_keywords else 0.0
        
        has_steps = any(w in response.lower() for w in ['step', 'first', 'then', 'next'])
        has_math = any(s in response for s in ['∫', '∂', '√', '∑', '∞', '=', '+', '-'])
        
        return {
            'problem': problem,
            'response': response,
            'keywords_found': keywords_found,
            'keyword_score': keyword_score,
            'has_steps': has_steps,
            'has_math': has_math,
            'type': problem_data.get('type', 'unknown'),
            'difficulty': problem_data.get('difficulty', 'unknown')
        }
    
    def evaluate_all(self, benchmark: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate all problems."""
        results = []
        
        for problem_data in tqdm(benchmark, desc="Evaluating"):
            result = self.evaluate_problem(problem_data)
            results.append(result)
        
        total_score = sum(r['keyword_score'] for r in results) / len(results)
        
        by_type = {}
        for r in results:
            ptype = r['type']
            if ptype not in by_type:
                by_type[ptype] = []
            by_type[ptype].append(r['keyword_score'])
        
        by_difficulty = {}
        for r in results:
            diff = r['difficulty']
            if diff not in by_difficulty:
                by_difficulty[diff] = []
            by_difficulty[diff].append(r['keyword_score'])
        
        return {
            'total_problems': len(results),
            'average_score': total_score,
            'by_type': {k: sum(v)/len(v) for k, v in by_type.items()},
            'by_difficulty': {k: sum(v)/len(v) for k, v in by_difficulty.items()},
            'has_steps_ratio': sum(1 for r in results if r['has_steps']) / len(results),
            'has_math_ratio': sum(1 for r in results if r['has_math']) / len(results),
            'detailed_results': results
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save results."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/evaluation_results.json")
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator()
    
    try:
        evaluator.load_model(args.model)
        benchmark = evaluator.load_benchmark(args.benchmark)
        results = evaluator.evaluate_all(benchmark)
        
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
        print("="*60 + "\n")
        
        evaluator.save_results(results, args.output)
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

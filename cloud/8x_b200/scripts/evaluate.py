#!/usr/bin/env python3
"""Evaluation Script for 8x_b200"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.get_full_config()
        self.model = None
        self.tokenizer = None
    
    def load_model(self, model_path: str):
        logger.info(f"Loading model from {model_path}")
        path = Path(model_path)
        
        self.tokenizer = AutoTokenizer.from_pretrained(str(path) if (path / "tokenizer.json").exists() else self.config['model']['base_model'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        base = AutoModelForCausalLM.from_pretrained(
            self.config['model']['base_model'], token=os.environ.get('HF_TOKEN'),
            quantization_config=bnb, torch_dtype=torch.float16
        )
        
        if (path / "adapter_config.json").exists():
            self.model = PeftModel.from_pretrained(base, str(path))
        else:
            self.model = base
    
    def load_benchmark(self):
        path = Path(__file__).parent.parent / "data" / "evaluation_benchmark.json"
        if path.exists():
            with open(path) as f: return json.load(f)
        return [
            {"problem": "Find derivative of sin(x)", "expected_keywords": ["cos(x)"], "type": "derivatives", "difficulty": "basic"},
            {"problem": "Compute integral of e^x dx", "expected_keywords": ["e^x", "+ C"], "type": "integrals", "difficulty": "basic"},
            {"problem": "Evaluate limit sin(x)/x as x->0", "expected_keywords": ["1"], "type": "limits", "difficulty": "basic"},
        ]
    
    def generate_response(self, problem: str) -> str:
        text = f"Problem: {problem}\nSolution:"
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=200, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)
        
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    
    def evaluate_all(self, benchmark):
        results = []
        for p in tqdm(benchmark, desc="Evaluating"):
            resp = self.generate_response(p['problem'])
            found = [kw for kw in p.get('expected_keywords', []) if kw.lower() in resp.lower()]
            score = len(found) / len(p['expected_keywords']) if p['expected_keywords'] else 0
            results.append({'problem': p['problem'], 'response': resp, 'score': score, 'type': p.get('type'), 'difficulty': p.get('difficulty')})
        
        avg = sum(r['score'] for r in results) / len(results)
        return {'total': len(results), 'average_score': avg, 'results': results}
    
    def save_results(self, results, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f: json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate 8x_b200")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/eval_results.json")
    args = parser.parse_args()
    
    evaluator = Evaluator()
    evaluator.load_model(args.model)
    benchmark = evaluator.load_benchmark()
    results = evaluator.evaluate_all(benchmark)
    
    print(f"\nResults: {results['total']} problems, avg score: {results['average_score']:.2%}")
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()

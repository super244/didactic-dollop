#!/usr/bin/env python3
"""Dataset Preparation Script for 8x_b200"""

import os
import sys
import json
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CalculusProblemGenerator:
    def __init__(self):
        self.problem_builders = {
            'derivatives': {'basic': self._basic_deriv, 'intermediate': self._inter_deriv, 'advanced': self._adv_deriv},
            'integrals': {'basic': self._basic_int, 'intermediate': self._inter_int, 'advanced': self._adv_int},
            'limits': {'basic': self._basic_lim, 'intermediate': self._inter_lim, 'advanced': self._adv_lim},
            'series': {'basic': self._basic_ser, 'intermediate': self._inter_ser, 'advanced': self._adv_ser},
            'applications': {'basic': self._basic_app, 'intermediate': self._inter_app, 'advanced': self._adv_app}
        }
    
    def _basic_deriv(self): return {"problem": "Find derivative of sin(x)", "solution": "cos(x)", "difficulty": "basic", "type": "derivatives"}
    def _inter_deriv(self): return {"problem": "Find derivative of sin(2x)", "solution": "2cos(2x)", "difficulty": "intermediate", "type": "derivatives"}
    def _adv_deriv(self): return {"problem": "Find dy/dx for x^2 + y^2 = 25", "solution": "dy/dx = -x/y", "difficulty": "advanced", "type": "derivatives"}
    def _basic_int(self): return {"problem": "Compute integral of e^x dx", "solution": "e^x + C", "difficulty": "basic", "type": "integrals"}
    def _inter_int(self): return {"problem": "Compute integral of x*sin(x) dx", "solution": "-x*cos(x) + sin(x) + C", "difficulty": "intermediate", "type": "integrals"}
    def _adv_int(self): return {"problem": "Compute integral of 1/(x^2 + 1) dx", "solution": "arctan(x) + C", "difficulty": "advanced", "type": "integrals"}
    def _basic_lim(self): return {"problem": "Evaluate limit of sin(x)/x as x->0", "solution": "1", "difficulty": "basic", "type": "limits"}
    def _inter_lim(self): return {"problem": "Evaluate limit of (x^2-4)/(x-2) as x->2", "solution": "4", "difficulty": "intermediate", "type": "limits"}
    def _adv_lim(self): return {"problem": "Evaluate limit of (1+1/x)^x as x->∞", "solution": "e", "difficulty": "advanced", "type": "limits"}
    def _basic_ser(self): return {"problem": "Write Maclaurin series for e^x", "solution": "1 + x + x^2/2 + x^3/6", "difficulty": "basic", "type": "series"}
    def _inter_ser(self): return {"problem": "Write Taylor series for e^x about x=1", "solution": "e + e(x-1) + e(x-1)^2/2", "difficulty": "intermediate", "type": "series"}
    def _adv_ser(self): return {"problem": "Find sum of Σ(n=1 to ∞) x^n/n", "solution": "ln(1/(1-x)), radius=1", "difficulty": "advanced", "type": "series"}
    def _basic_app(self): return {"problem": "Ball thrown at 20 m/s. Find max height.", "solution": "20.4 meters", "difficulty": "basic", "type": "applications"}
    def _inter_app(self): return {"problem": "Find volume of y=√x rotated about x-axis", "solution": "8π cubic units", "difficulty": "intermediate", "type": "applications"}
    def _adv_app(self): return {"problem": "Tank drains at rate dh/dt = -0.1√h. Time to empty?", "solution": "40/3 seconds", "difficulty": "advanced", "type": "applications"}
    
    def generate_problem(self, ptype: str, diff: str) -> Dict[str, str]:
        return self.problem_builders[ptype][diff]()


class DatasetPreparer:
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.get_full_config()
        self.dataset_config = self.config_loader.get_dataset_config()
        self.generator = CalculusProblemGenerator()
    
    def generate_dataset(self) -> List[Dict[str, str]]:
        size = self.dataset_config['size']
        problem_dist = self.dataset_config['problem_distribution']
        difficulty_prog = self.dataset_config['difficulty_progression']
        
        logger.info(f"Generating {size} problems for 8x_b200")
        problems = []
        
        for ptype, ratio in problem_dist.items():
            type_count = int(size * ratio)
            for diff, dratio in difficulty_prog.items():
                dcount = int(type_count * dratio)
                for _ in range(dcount):
                    try:
                        problems.append(self.generator.generate_problem(ptype, diff))
                    except: continue
        
        random.shuffle(problems)
        return problems[:size]
    
    def save_dataset(self, problems: List[Dict[str, str]]):
        output_dir = Path(__file__).parent.parent / "data"
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "calculus_problems.jsonl", 'w') as f:
            for p in problems: f.write(json.dumps(p) + '\n')
        
        logger.info(f"Dataset saved to {output_dir}")
    
    def prepare(self):
        problems = self.generate_dataset()
        self.save_dataset(problems)


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for 8x_b200")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    dataset_path = Path(__file__).parent.parent / "data" / "calculus_problems.jsonl"
    if dataset_path.exists() and not args.overwrite:
        logger.info(f"Dataset exists: {dataset_path}")
        return 0
    
    preparer = DatasetPreparer()
    preparer.prepare()
    return 0


if __name__ == "__main__":
    sys.exit(main())

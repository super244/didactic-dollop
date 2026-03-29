#!/usr/bin/env python3
"""
Dataset Preparation Script for macOS
Generates calculus problems optimized for local training.
"""

import os
import sys
import json
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config_loader import ConfigLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CalculusProblemGenerator:
    """Advanced calculus problem generator with multiple difficulty levels and types."""
    
    def __init__(self):
        self.problem_builders = {
            'derivatives': {
                'basic': self._build_basic_derivative,
                'intermediate': self._build_intermediate_derivative,
                'advanced': self._build_advanced_derivative
            },
            'integrals': {
                'basic': self._build_basic_integral,
                'intermediate': self._build_intermediate_integral,
                'advanced': self._build_advanced_integral
            },
            'limits': {
                'basic': self._build_basic_limit,
                'intermediate': self._build_intermediate_limit,
                'advanced': self._build_advanced_limit
            },
            'series': {
                'basic': self._build_basic_series,
                'intermediate': self._build_intermediate_series,
                'advanced': self._build_advanced_series
            },
            'applications': {
                'basic': self._build_basic_application,
                'intermediate': self._build_intermediate_application,
                'advanced': self._build_advanced_application
            }
        }
    
    def _build_basic_derivative(self) -> Dict[str, str]:
        """Basic derivative problems."""
        power = random.randint(2, 5)
        expressions = [
            ("sin(x)", "cos(x)"),
            ("cos(x)", "-sin(x)"),
            ("e^x", "e^x"),
            ("ln(x)", "1/x"),
            (f"x^{power}", f"{power}x^{power - 1}"),
        ]
        expression, derivative = random.choice(expressions)
        return {
            "problem": f"Find the derivative of {expression} with respect to x.",
            "solution": f"d/dx {expression} = {derivative}",
            "difficulty": "basic",
            "type": "derivatives"
        }
    
    def _build_intermediate_derivative(self) -> Dict[str, str]:
        """Intermediate derivative problems (chain rule, product rule)."""
        problems = [
            ("sin(2x)", "2cos(2x)", "chain rule"),
            ("e^(x^2)", "2xe^(x^2)", "chain rule"),
            ("x^2 * sin(x)", "2x*sin(x) + x^2*cos(x)", "product rule"),
            ("ln(x^2 + 1)", "2x/(x^2 + 1)", "chain rule"),
            ("(sin(x))^2", "2*sin(x)*cos(x)", "chain rule"),
        ]
        expression, derivative, method = random.choice(problems)
        return {
            "problem": f"Find the derivative of {expression} using {method}.",
            "solution": f"d/dx {expression} = {derivative}",
            "difficulty": "intermediate",
            "type": "derivatives"
        }
    
    def _build_advanced_derivative(self) -> Dict[str, str]:
        """Advanced derivative problems (implicit differentiation, partial derivatives)."""
        problems = [
            ("x^2 + y^2 = 25", "dy/dx = -x/y", "implicit differentiation"),
            ("x^3 + y^3 = 6xy", "dy/dx = (6y - 3x^2)/(3y^2 - 6x)", "implicit differentiation"),
            ("f(x,y) = x^2*y + sin(xy)", "∂f/∂x = 2xy + y*cos(xy), ∂f/∂y = x^2 + x*cos(xy)", "partial derivatives"),
            ("y = tan^(-1)(x)", "dy/dx = 1/(1 + x^2)", "inverse trigonometric"),
        ]
        expression, derivative, method = random.choice(problems)
        return {
            "problem": f"Find the derivative(s) for: {expression}",
            "solution": derivative,
            "difficulty": "advanced",
            "type": "derivatives"
        }
    
    def _build_basic_integral(self) -> Dict[str, str]:
        """Basic integral problems."""
        power = random.randint(1, 4)
        expressions = [
            ("sin(x)", "-cos(x) + C"),
            ("cos(x)", "sin(x) + C"),
            ("e^x", "e^x + C"),
            (f"x^{power}", f"x^{power + 1}/{power + 1} + C"),
            ("1/x", "ln|x| + C"),
        ]
        expression, integral = random.choice(expressions)
        return {
            "problem": f"Compute the integral of {expression} dx.",
            "solution": f"∫ {expression} dx = {integral}",
            "difficulty": "basic",
            "type": "integrals"
        }
    
    def _build_intermediate_integral(self) -> Dict[str, str]:
        """Intermediate integral problems (substitution, integration by parts)."""
        problems = [
            ("x*e^(x^2)", "0.5*e^(x^2) + C", "u-substitution"),
            ("x*sin(x)", "-x*cos(x) + sin(x) + C", "integration by parts"),
            ("ln(x)", "x*ln(x) - x + C", "integration by parts"),
            ("sin(2x)", "-0.5*cos(2x) + C", "u-substitution"),
        ]
        expression, integral, method = random.choice(problems)
        return {
            "problem": f"Compute the integral of {expression} dx using {method}.",
            "solution": f"∫ {expression} dx = {integral}",
            "difficulty": "intermediate",
            "type": "integrals"
        }
    
    def _build_advanced_integral(self) -> Dict[str, str]:
        """Advanced integral problems (trig substitution, partial fractions)."""
        problems = [
            ("1/(x^2 + 1)", "arctan(x) + C", "standard form"),
            ("1/√(x^2 + 1)", "ln|x + √(x^2 + 1)| + C", "hyperbolic trig"),
            ("1/(x^2 - 4)", "0.25*ln|(x-2)/(x+2)| + C", "partial fractions"),
            ("e^x*sin(x)", "0.5*e^x*(sin(x) - cos(x)) + C", "integration by parts twice"),
        ]
        expression, integral, method = random.choice(problems)
        return {
            "problem": f"Compute the integral of {expression} dx.",
            "solution": f"∫ {expression} dx = {integral}",
            "difficulty": "advanced",
            "type": "integrals"
        }
    
    def _build_basic_limit(self) -> Dict[str, str]:
        """Basic limit problems."""
        problems = [
            ("sin(x)/x as x -> 0", "1"),
            ("(1 - cos(x))/x^2 as x -> 0", "1/2"),
            ("(e^x - 1)/x as x -> 0", "1"),
            ("x^2 as x -> 3", "9"),
            ("1/x as x -> ∞", "0"),
        ]
        expression, limit_value = random.choice(problems)
        return {
            "problem": f"Evaluate the limit of {expression}.",
            "solution": f"The limit is {limit_value}.",
            "difficulty": "basic",
            "type": "limits"
        }
    
    def _build_intermediate_limit(self) -> Dict[str, str]:
        """Intermediate limit problems (L'Hôpital's rule)."""
        problems = [
            ("(x^2 - 4)/(x - 2) as x -> 2", "4", "factoring"),
            ("sin(3x)/x as x -> 0", "3", "standard limit"),
            ("(e^x - 1 - x)/x^2 as x -> 0", "1/2", "Taylor series"),
            ("x*ln(x) as x -> 0+", "0", "special case"),
        ]
        expression, limit_value, method = random.choice(problems)
        return {
            "problem": f"Evaluate the limit of {expression} using {method}.",
            "solution": f"The limit is {limit_value}.",
            "difficulty": "intermediate",
            "type": "limits"
        }
    
    def _build_advanced_limit(self) -> Dict[str, str]:
        """Advanced limit problems (indeterminate forms, series)."""
        problems = [
            ("(1 + 1/x)^x as x -> ∞", "e", "definition of e"),
            ("x^(1/x) as x -> ∞", "1", "logarithmic approach"),
            ("(sin(x))/x as x -> ∞", "0", "bounded over infinity"),
            ("(x^2 + sin(x))/x^2 as x -> ∞", "1", "dominant term"),
        ]
        expression, limit_value, method = random.choice(problems)
        return {
            "problem": f"Evaluate the limit of {expression}.",
            "solution": f"The limit is {limit_value}.",
            "difficulty": "advanced",
            "type": "limits"
        }
    
    def _build_basic_series(self) -> Dict[str, str]:
        """Basic series problems (Maclaurin series)."""
        problems = [
            ("e^x up to the x^3 term", "1 + x + x^2/2 + x^3/6"),
            ("sin(x) up to the x^5 term", "x - x^3/6 + x^5/120"),
            ("cos(x) up to the x^4 term", "1 - x^2/2 + x^4/24"),
            ("ln(1 + x) up to the x^3 term", "x - x^2/2 + x^3/3"),
        ]
        expression, expansion = random.choice(problems)
        return {
            "problem": f"Write the Maclaurin series for {expression}.",
            "solution": expansion,
            "difficulty": "basic",
            "type": "series"
        }
    
    def _build_intermediate_series(self) -> Dict[str, str]:
        """Intermediate series problems (Taylor series, convergence)."""
        problems = [
            ("e^x about x = 1 up to (x-1)^3 term", "e + e(x-1) + e(x-1)^2/2 + e(x-1)^3/6"),
            ("sin(x) about x = π/2 up to (x-π/2)^3 term", "1 - (x-π/2)^2/2"),
            ("1/(1-x) for |x| < 1", "1 + x + x^2 + x^3 + ..."),
            ("ln(x) about x = 1 up to (x-1)^3 term", "(x-1) - (x-1)^2/2 + (x-1)^3/3"),
        ]
        expression, expansion = random.choice(problems)
        return {
            "problem": f"Write the Taylor series for {expression}.",
            "solution": expansion,
            "difficulty": "intermediate",
            "type": "series"
        }
    
    def _build_advanced_series(self) -> Dict[str, str]:
        """Advanced series problems (radius of convergence, power series)."""
        problems = [
            ("Σ(n=1 to ∞) x^n/n", "ln(1/(1-x)), radius of convergence: 1"),
            ("Σ(n=0 to ∞) (-1)^n x^(2n)/(2n)!", "cos(x), radius of convergence: ∞"),
            ("Σ(n=1 to ∞) n x^n", "x/(1-x)^2, radius of convergence: 1"),
            ("Σ(n=0 to ∞) x^(2n+1)/(2n+1)!", "sinh(x), radius of convergence: ∞"),
        ]
        expression, expansion = random.choice(problems)
        return {
            "problem": f"Find the sum and radius of convergence for: {expression}",
            "solution": expansion,
            "difficulty": "advanced",
            "type": "series"
        }
    
    def _build_basic_application(self) -> Dict[str, str]:
        """Basic application problems."""
        problems = [
            ("A ball is thrown upward with velocity 20 m/s. Find its maximum height.", "Maximum height is 20.4 meters"),
            ("Find the area under y = x^2 from x = 0 to x = 2", "Area is 8/3 square units"),
            ("A rectangle has length x and width 10-x. Find maximum area.", "Maximum area is 25 square units"),
        ]
        problem, solution = random.choice(problems)
        return {
            "problem": problem,
            "solution": solution,
            "difficulty": "basic",
            "type": "applications"
        }
    
    def _build_intermediate_application(self) -> Dict[str, str]:
        """Intermediate application problems."""
        problems = [
            ("A particle moves with position s(t) = t^3 - 6t^2 + 9t. Find when velocity is zero.", "Velocity is zero at t = 1 and t = 3"),
            ("Find the volume of revolution for y = √x from x = 0 to x = 4 about the x-axis.", "Volume is 8π cubic units"),
            ("A company's profit is P(x) = -x^2 + 100x - 1000. Find maximum profit.", "Maximum profit is $1500 at x = 50"),
        ]
        problem, solution = random.choice(problems)
        return {
            "problem": problem,
            "solution": solution,
            "difficulty": "intermediate",
            "type": "applications"
        }
    
    def _build_advanced_application(self) -> Dict[str, str]:
        """Advanced application problems."""
        problems = [
            ("A tank drains with rate dh/dt = -0.1√h. Find time to empty from h=4 to h=0.", "Time is 40/3 seconds"),
            ("Find the surface area of y = x^2 from x = 0 to x = 1 rotated about the x-axis.", "Surface area is approximately 5.33 square units"),
            ("A population grows according to dP/dt = 0.02P(1 - P/1000). Find equilibrium points.", "Equilibrium at P = 0 and P = 1000"),
        ]
        problem, solution = random.choice(problems)
        return {
            "problem": problem,
            "solution": solution,
            "difficulty": "advanced",
            "type": "applications"
        }
    
    def generate_problem(self, problem_type: str, difficulty: str) -> Dict[str, str]:
        """Generate a single problem of given type and difficulty."""
        if problem_type not in self.problem_builders:
            raise ValueError(f"Unknown problem type: {problem_type}")
        
        if difficulty not in self.problem_builders[problem_type]:
            raise ValueError(f"Unknown difficulty level: {difficulty}")
        
        return self.problem_builders[problem_type][difficulty]()


class DatasetPreparer:
    """Dataset preparation for macOS training."""
    
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.get_full_config()
        self.dataset_config = self.config_loader.get_dataset_config()
        self.generator = CalculusProblemGenerator()
        
    def generate_dataset(self) -> List[Dict[str, str]]:
        """Generate dataset based on configuration."""
        size = self.dataset_config['size']
        problem_dist = self.dataset_config['problem_distribution']
        difficulty_prog = self.dataset_config['difficulty_progression']
        
        logger.info(f"Generating {size} problems for macOS MLX training")
        
        problems = []
        
        # Calculate number of problems for each type and difficulty
        for problem_type, type_ratio in problem_dist.items():
            type_count = int(size * type_ratio)
            
            for difficulty, diff_ratio in difficulty_prog.items():
                diff_count = int(type_count * diff_ratio)
                
                for _ in range(diff_count):
                    try:
                        problem = self.generator.generate_problem(problem_type, difficulty)
                        problems.append(problem)
                    except Exception as e:
                        logger.warning(f"Failed to generate {problem_type} {difficulty} problem: {e}")
                        continue
        
        # Shuffle the dataset
        random.shuffle(problems)
        
        # Trim to exact size if needed
        if len(problems) > size:
            problems = problems[:size]
        
        logger.info(f"Generated {len(problems)} problems")
        return problems
    
    def save_dataset(self, problems: List[Dict[str, str]]):
        """Save dataset to file."""
        output_dir = Path(__file__).parent.parent / "data"
        output_dir.mkdir(exist_ok=True)
        
        dataset_path = output_dir / "calculus_problems.jsonl"
        
        with open(dataset_path, 'w', encoding='utf-8') as f:
            for problem in problems:
                f.write(json.dumps(problem) + '\n')
        
        logger.info(f"Dataset saved to {dataset_path}")
        
        # Save metadata
        metadata = {
            'config_name': 'macos_mlx',
            'total_problems': len(problems),
            'problem_distribution': self.dataset_config['problem_distribution'],
            'difficulty_progression': self.dataset_config['difficulty_progression'],
            'generated_at': str(Path.cwd())
        }
        
        metadata_path = output_dir / "calculus_problems_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
    
    def prepare(self):
        """Main preparation function."""
        logger.info("Preparing dataset for macOS MLX training")
        
        # Generate problems
        problems = self.generate_dataset()
        
        # Save dataset
        self.save_dataset(problems)
        
        logger.info("Dataset preparation completed!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Prepare calculus dataset for macOS training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing dataset")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Check if dataset already exists
    dataset_path = Path(__file__).parent.parent / "data" / "calculus_problems.jsonl"
    if dataset_path.exists() and not args.overwrite:
        logger.info(f"Dataset already exists: {dataset_path}")
        logger.info("Use --overwrite to regenerate")
        return 0
    
    # Initialize preparer
    try:
        preparer = DatasetPreparer()
        
        # Prepare dataset
        preparer.prepare()
        return 0
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

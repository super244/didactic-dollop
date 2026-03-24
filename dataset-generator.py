# dataset-generator.py

# imports
import random
import json
import re

# generate calculus problems
def generate_calculus_problems(n=1000):
    """Generate synthetic calculus problems with solutions"""
    problems = []
    for _ in range(n):
        problem_type = random.choice(["derivative", "integral", "limit", "series"])
        x = random.uniform(0.1, 10.0)

        if problem_type == "derivative":
            expr = f"sin({x})", "cos({x})", "e^{x}", "ln({x})"
            expr = random.choice(expr)
            problem = f"Find the derivative of {expr} with respect to x."
            solution = f"d/dx {expr} = {re.sub(r'([a-z])', r'\1)', expr)}"
        elif problem_type == "integral":
            expr = f"x^{random.randint(2,5)}", "sin({x})", "e^{x}"
            expr = random.choice(expr)
            problem = f"Compute the integral of {expr} dx."
            solution = f"∫{expr} dx = {re.sub(r'([a-z])', r'\1)', expr)}"
        # Add more types as needed
        
        problems.append({
            "problem": problem,
            "solution": solution
        })
    
    with open("data/calculus_problems.jsonl", "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")
    print(f"✅ Generated {n} calculus problems to data/calculus_problems.jsonl")

if __name__ == "__main__":
    generate_calculus_problems()
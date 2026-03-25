import json
from pathlib import Path
import random


def build_derivative_example():
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
    }


def build_integral_example():
    power = random.randint(1, 4)
    expressions = [
        ("sin(x)", "-cos(x) + C"),
        ("cos(x)", "sin(x) + C"),
        ("e^x", "e^x + C"),
        (f"x^{power}", f"x^{power + 1}/{power + 1} + C"),
    ]
    expression, integral = random.choice(expressions)
    return {
        "problem": f"Compute the integral of {expression} dx.",
        "solution": f"∫ {expression} dx = {integral}",
    }


def build_limit_example():
    expressions = [
        ("sin(x)/x as x -> 0", "1"),
        ("(1 - cos(x))/x^2 as x -> 0", "1/2"),
        ("(e^x - 1)/x as x -> 0", "1"),
    ]
    expression, limit_value = random.choice(expressions)
    return {
        "problem": f"Evaluate the limit of {expression}.",
        "solution": f"The limit is {limit_value}.",
    }


def build_series_example():
    series = [
        ("e^x up to the x^3 term", "1 + x + x^2/2 + x^3/6"),
        ("sin(x) up to the x^5 term", "x - x^3/6 + x^5/120"),
        ("cos(x) up to the x^4 term", "1 - x^2/2 + x^4/24"),
    ]
    expression, expansion = random.choice(series)
    return {
        "problem": f"Write the Maclaurin series for {expression}.",
        "solution": expansion,
    }


def generate_calculus_problems(n=1000):
    """Generate synthetic calculus problems with simple reference solutions."""
    builders = [
        build_derivative_example,
        build_integral_example,
        build_limit_example,
        build_series_example,
    ]
    problems = [random.choice(builders)() for _ in range(n)]
    Path("data").mkdir(parents=True, exist_ok=True)

    with open("data/calculus_problems.jsonl", "w", encoding="utf-8") as handle:
        for problem in problems:
            handle.write(json.dumps(problem) + "\n")

    print(f"Generated {n} calculus problems to data/calculus_problems.jsonl")


if __name__ == "__main__":
    generate_calculus_problems()

import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Base vs Fine-tuned Model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model name or path")
    parser.add_argument("--adapter_path", type=str, default="qwen-calculus-finetuned", help="Path to the LoRA adapter")
    parser.add_argument("--config_path", type=str, default="eval_configs.json", help="Path to evaluation configs JSON")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Max new tokens to generate")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"), help="HuggingFace Token")
    return parser.parse_args()

def generate_response(model, tokenizer, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to model device
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

def main():
    args = parse_args()

    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")

    with open(args.config_path, "r") as f:
        configs = json.load(f)

    use_cuda = torch.cuda.is_available()
    device_map = "auto" if use_cuda else None
    torch_dtype = torch.float16 if use_cuda else torch.float32

    print(f"Loading base model: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token=args.hf_token,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    
    if os.path.exists(args.adapter_path):
        print(f"Loading adapter from: {args.adapter_path}...")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
    else:
        print(f"Warning: Adapter path '{args.adapter_path}' not found. Only evaluating base model.")
        model = base_model

    print("\n" + "="*50)
    print("Starting Evaluation")
    print("="*50 + "\n")

    for idx, config in enumerate(configs):
        problem = config["problem"]
        prompt = f"Problem: {problem}\nSolution:"
        
        print(f"--- Example {idx + 1} ---")
        print(f"Prompt: {problem}\n")
        
        # Base Model Generation
        if isinstance(model, PeftModel):
            with model.disable_adapter():
                base_response = generate_response(model, tokenizer, prompt, args.max_new_tokens)
        else:
            base_response = generate_response(model, tokenizer, prompt, args.max_new_tokens)
            
        print("[Base Model]:")
        print(base_response)
        print("-" * 30)
        
        # Fine-tuned Model Generation
        if isinstance(model, PeftModel):
            tuned_response = generate_response(model, tokenizer, prompt, args.max_new_tokens)
            print("[Fine-tuned Model]:")
            print(tuned_response)
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()

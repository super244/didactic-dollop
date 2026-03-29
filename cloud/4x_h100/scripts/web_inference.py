#!/usr/bin/env python3
"""Web Inference Script for 4x_h100"""

import os
import sys
import glob
import torch
import gradio as gr
from pathlib import Path
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from scripts.config_loader import ConfigLoader

BASE_MODEL = os.environ.get("BASE_MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN")

tokenizer = None
base_model = None
model = None


class Tutor:
    def __init__(self):
        self.load_base_model()
        self.adapters = self.get_adapters()
    
    def load_base_model(self):
        global tokenizer, base_model, model
        
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, token=HF_TOKEN, quantization_config=bnb, torch_dtype=torch.float16)
        model = base_model
        print("Base model loaded")
    
    def load_adapter(self, path: str) -> str:
        global model
        if not path or path == "None":
            model = base_model
            return "Using base model"
        
        if not os.path.exists(path): return f"Error: {path} not found"
        
        try:
            model = PeftModel.from_pretrained(base_model, path)
            return f"Loaded: {path}"
        except Exception as e:
            return f"Error: {e}"
    
    def get_adapters(self) -> List[str]:
        adapters = ["None"]
        output_dir = Path(__file__).parent.parent / "outputs"
        if output_dir.exists():
            adapters.extend(sorted(glob.glob(str(output_dir / "checkpoint-*"))))
            adapters.extend(sorted(glob.glob(str(output_dir / "fine_tuned_iteration_*"))))
        return adapters
    
    def generate(self, message: str, history: List, temp: float, top_p: float, max_tok: int) -> str:
        if not message.strip(): return "Enter a question"
        
        msgs = [{"role": "system", "content": "You are an expert calculus tutor."}]
        for u, a in history:
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user", "content": message})
        
        try:
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except:
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in msgs]) + "\nAssistant:"
        
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=max_tok, temperature=temp if temp > 0 else 1.0, top_p=top_p, do_sample=temp > 0, pad_token_id=tokenizer.eos_token_id)
        
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def create_interface():
    tutor = Tutor()
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.HTML("""<div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1>Calculus Tutor - """ + config_name.replace('_', ' ').upper() + """</h1>
            <p>AI-powered calculus tutoring</p>
        </div>""")
        
        with gr.Row():
            with gr.Column(scale=1):
                adapter = gr.Dropdown(choices=tutor.adapters, value="None", label="Model")
                load_btn = gr.Button("Load Model", variant="primary")
                status = gr.Markdown("")
                temp = gr.Slider(0, 2, 0.7, label="Temperature")
                top_p = gr.Slider(0, 1, 0.9, label="Top-P")
                max_tok = gr.Slider(50, 1024, 512, label="Max Tokens")
            
            with gr.Column(scale=2):
                chat = gr.ChatInterface(fn=tutor.generate, additional_inputs=[temp, top_p, max_tok], title="Chat")
        
        load_btn.click(fn=tutor.load_adapter, inputs=[adapter], outputs=[status])
        gr.Markdown("---
**" + config_name.replace('_', ' ').upper() + " Calculus Tutor**")
    
    return demo


def main():
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)


if __name__ == "__main__":
    main()

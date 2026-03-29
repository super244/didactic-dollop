#!/usr/bin/env python3
"""
Web Inference Script for RTX 5090
Gradio-based web interface for model inference.
"""

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

DEFAULT_BASE_MODEL = os.environ.get("BASE_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN")

tokenizer = None
base_model = None
model = None
current_adapter = None
config_loader = ConfigLoader()


class CalculusTutorWeb:
    """Web interface for RTX 5090."""
    
    def __init__(self):
        self.load_base_model()
        self.available_adapters = self.get_available_adapters()
        
    def load_base_model(self):
        """Load base model."""
        global tokenizer, base_model, model
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL, token=HF_TOKEN)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_BASE_MODEL,
            token=HF_TOKEN,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
        )
        model = base_model
        print("Base model loaded successfully")
    
    def load_adapter(self, adapter_path: str) -> str:
        """Load adapter."""
        global model, current_adapter
        
        if not adapter_path or adapter_path == "None":
            model = base_model
            current_adapter = None
            return "Unloaded adapter. Using base model."
        
        if not os.path.exists(adapter_path):
            return f"Error: Adapter path {adapter_path} does not exist."
        
        try:
            model = PeftModel.from_pretrained(base_model, adapter_path)
            current_adapter = adapter_path
            return f"Successfully loaded adapter: {adapter_path}"
        except Exception as e:
            return f"Error loading adapter: {str(e)}"
    
    def get_available_adapters(self) -> List[str]:
        """Get available adapters."""
        adapters = ["None"]
        
        output_dir = Path(__file__).parent.parent / "outputs"
        if output_dir.exists():
            checkpoints = sorted(glob.glob(str(output_dir / "checkpoint-*")))
            adapters.extend(checkpoints)
            
            fine_tuned = sorted(glob.glob(str(output_dir / "fine_tuned_iteration_*")))
            adapters.extend(fine_tuned)
            
            best_model = output_dir / "best_model"
            if best_model.exists():
                adapters.append(str(best_model))
        
        return adapters
    
    def generate_response(self, message: str, history: List, use_system_prompt: bool,
                         system_prompt: str, temperature: float, top_p: float,
                         max_tokens: int) -> str:
        """Generate response."""
        if not message.strip():
            return "Enter a calculus question."
        
        messages = []
        if use_system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        
        messages.append({"role": "user", "content": message})
        
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = ""
            for msg in messages:
                prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
            prompt += "Assistant:"
        
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0.0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0.0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()


def create_interface():
    """Create Gradio interface."""
    tutor = CalculusTutorWeb()
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1>🧮 Calculus Tutor - RTX 5090</h1>
            <p>AI-powered calculus tutoring trained on consumer GPU</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Model Configuration")
                adapter_dropdown = gr.Dropdown(choices=tutor.available_adapters, value="None", label="Select Model/Adapter")
                load_btn = gr.Button("Load Model", variant="primary")
                load_status = gr.Markdown("")
                
                gr.Markdown("### Generation Settings")
                use_system_prompt = gr.Checkbox(label="Enable System Prompt", value=True)
                system_prompt = gr.Textbox(value="You are an expert calculus tutor. Provide step-by-step solutions.", label="System Prompt", lines=3)
                temperature = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature")
                top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Top-P")
                max_tokens = gr.Slider(minimum=50, maximum=1024, value=512, step=50, label="Max Tokens")
            
            with gr.Column(scale=2):
                chatbot = gr.ChatInterface(
                    fn=tutor.generate_response,
                    additional_inputs=[use_system_prompt, system_prompt, temperature, top_p, max_tokens],
                    title="Calculus Tutor Chat"
                )
        
        load_btn.click(fn=tutor.load_adapter, inputs=[adapter_dropdown], outputs=[load_status])
        
        gr.Markdown("---\n**RTX 5090 Calculus Tutor** | 24GB VRAM Consumer GPU")
    
    return demo


def main():
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)


if __name__ == "__main__":
    main()

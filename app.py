# app.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# Load fine-tuned model
model_path = "qwen3-calculus-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16
)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Gradio UI
gr.Interface(
    fn=generate_response,
    inputs="text",
    outputs="text",
    title="Calculus Tutor (Qwen3-7B-Instruct)",
    description="Ask calculus questions. Powered by fine-tuned Qwen3-7B-Instruct.",
    examples=["Find derivative of sin(x)", "Integrate x^2 dx", "Limit of (sin(x)/x) as x->0"]
).launch(server_name="0.0.0.0", server_port=7860)

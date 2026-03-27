import os
import glob
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
DEFAULT_BASE_MODEL = os.environ.get("BASE_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DEFAULT_ADAPTER_DIR = "qwen-calculus-finetuned"
HF_TOKEN = os.environ.get("HF_TOKEN")

tokenizer = None
base_model = None
model = None
current_adapter = None

def load_base_model():
    global tokenizer, base_model, model
    use_cuda = torch.cuda.is_available()
    try:
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL, token=HF_TOKEN)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        base_model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_BASE_MODEL,
            token=HF_TOKEN,
            device_map="auto" if use_cuda else None,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
        )
        model = base_model
    except OSError as exc:
        raise RuntimeError(
            f"Unable to load base model {DEFAULT_BASE_MODEL}. Verify BASE_MODEL_NAME or HF_TOKEN."
        ) from exc

def load_adapter(adapter_path):
    global base_model, model, current_adapter
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

# Initialize models
load_base_model()
if os.path.exists(DEFAULT_ADAPTER_DIR):
    load_adapter(DEFAULT_ADAPTER_DIR)

def generate_response(message, history, use_system_prompt, system_prompt, temperature, top_p):
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
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    except Exception:
        # Fallback if chat template is not available
        prompt = ""
        for msg in messages:
            prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
        prompt += "Assistant:"

    inputs = tokenizer(prompt, return_tensors="pt")
    if not hasattr(model, "hf_device_map"):
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0.0
        )
    
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return response

def get_available_adapters():
    adapters = ["None"]
    if os.path.exists(DEFAULT_ADAPTER_DIR):
        adapters.append(DEFAULT_ADAPTER_DIR)
        checkpoints = sorted(glob.glob(f"{DEFAULT_ADAPTER_DIR}/checkpoint-*"))
        adapters.extend(checkpoints)
    return adapters

def main():
    css = "body { font-family: monospace; }"
    
    with gr.Blocks(theme=gr.themes.Monochrome(), css=css) as demo:
        gr.Markdown("# Calculus Tutor (Tuning Station)")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Adapter Configuration")
                adapter_dropdown = gr.Dropdown(
                    choices=get_available_adapters(), 
                    value=current_adapter if current_adapter else "None",
                    label="Select Adapter"
                )
                load_btn = gr.Button("Load Adapter", variant="primary")
                load_status = gr.Markdown("")
                
            with gr.Column(scale=3):
                gr.ChatInterface(
                    fn=generate_response,
                    additional_inputs=[
                        gr.Checkbox(label="Enable System Prompt", value=True),
                        gr.Textbox(
                            value="You are an expert calculus tutor. Provide step-by-step solutions.",
                            label="System Prompt",
                            lines=3
                        ),
                        gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
                        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Top-P"),
                    ]
                )
        
        load_btn.click(
            fn=load_adapter,
            inputs=[adapter_dropdown],
            outputs=[load_status]
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()

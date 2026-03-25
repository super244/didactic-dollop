import os

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = "qwen-calculus-finetuned"
FALLBACK_MODEL_NAME = os.environ.get("BASE_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN")


def resolve_model_path():
    configured_path = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
    if os.path.exists(configured_path):
        return configured_path
    return os.environ.get("BASE_MODEL_NAME", FALLBACK_MODEL_NAME)


def load_model():
    model_path = resolve_model_path()
    use_cuda = torch.cuda.is_available()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=HF_TOKEN,
            device_map="auto" if use_cuda else None,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
        )
    except OSError as exc:
        raise RuntimeError(
            "Unable to load a model for the app. Train the repo first, or set BASE_MODEL_NAME "
            "to a valid public model. Use HF_TOKEN or `hf auth login` for gated/private repos."
        ) from exc
    return tokenizer, model


tokenizer, model = load_model()


def generate_response(prompt):
    if not prompt or not prompt.strip():
        return "Enter a calculus question."

    inputs = tokenizer(prompt, return_tensors="pt")
    if not hasattr(model, "hf_device_map"):
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=200)
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def main():
    gr.Interface(
        fn=generate_response,
        inputs="text",
        outputs="text",
        title="Calculus Tutor",
        description="Ask calculus questions. Powered by a fine-tuned Qwen model when available.",
        examples=[
            "Find the derivative of sin(x)",
            "Integrate x^2 dx",
            "Limit of sin(x)/x as x approaches 0",
        ],
    ).launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()

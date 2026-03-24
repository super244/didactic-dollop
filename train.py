# train.py
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset

# Configuration
MODEL_NAME = "Qwen/Qwen3-7B-Instruct"  # Base model (not Qwen3.5)
DATASET_PATH = "data/calculus_problems.jsonl"
OUTPUT_DIR = "qwen3-calculus-finetuned"

# Load dataset
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# Tokenize
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize_function(examples):
    return tokenizer(
        examples["problem"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)
model = get_peft_model(model, lora_config)

# Training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    fp16=True,
    output_dir=OUTPUT_DIR,
    num_cpu_threads=4,
    report_to="none"
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# Save final model
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Fine-tuning complete! Model saved to {OUTPUT_DIR}")

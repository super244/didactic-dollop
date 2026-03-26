import inspect
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_NAME = os.environ.get("BASE_MODEL_NAME", DEFAULT_BASE_MODEL)
DATASET_PATH = "data/calculus_problems.jsonl"
OUTPUT_DIR = "qwen-calculus-finetuned"
MAX_LENGTH = 512
HF_TOKEN = os.environ.get("HF_TOKEN")


def format_examples(problems, solutions):
    return [
        f"Problem: {problem}\nSolution: {solution}"
        for problem, solution in zip(problems, solutions)
    ]


def build_training_args(use_cuda):
    supported_args = inspect.signature(TrainingArguments.__init__).parameters
    training_kwargs = {
        "output_dir": OUTPUT_DIR,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "fp16": use_cuda,
        "report_to": "none",
        "remove_unused_columns": False,
    }

    if "evaluation_strategy" in supported_args:
        training_kwargs["evaluation_strategy"] = "no"
    elif "eval_strategy" in supported_args:
        training_kwargs["eval_strategy"] = "no"

    return TrainingArguments(**training_kwargs)


def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    except OSError as exc:
        raise RuntimeError(
            "Unable to load the base model. Set BASE_MODEL_NAME to a valid public model "
            f"or authenticate with HF_TOKEN if the repo is private or gated. Current value: {MODEL_NAME}"
        ) from exc
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            format_examples(examples["problem"], examples["solution"]),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    use_cuda = torch.cuda.is_available()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            token=HF_TOKEN,
            device_map="auto" if use_cuda else None,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
        )
    except OSError as exc:
        raise RuntimeError(
            "Unable to download the base model weights. Verify BASE_MODEL_NAME, run `hf auth login`, "
            "or export HF_TOKEN for gated/private repositories."
        ) from exc
    model.config.pad_token_id = tokenizer.pad_token_id
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    training_args = build_training_args(use_cuda)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Fine-tuning complete. Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

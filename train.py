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


MODEL_NAME = "Qwen/Qwen3-7B-Instruct"
DATASET_PATH = "data/calculus_problems.jsonl"
OUTPUT_DIR = "qwen3-calculus-finetuned"
MAX_LENGTH = 512


def format_examples(problems, solutions):
    return [
        f"Problem: {problem}\nSolution: {solution}"
        for problem, solution in zip(problems, solutions)
    ]


def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto" if use_cuda else None,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
    )
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
        fp16=use_cuda,
        report_to="none",
        remove_unused_columns=False,
    )

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

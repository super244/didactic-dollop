import argparse
import os
import torch
import logging
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a LoRA/QLoRA model.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model name or path")
    parser.add_argument("--dataset_path", type=str, default="data/calculus_problems.jsonl", help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default="qwen-calculus-finetuned", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--grad_acc_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--use_qlora", action="store_true", help="Use QLoRA (4-bit quantization)")
    parser.add_argument("--push_to_hub", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"), help="HuggingFace Token")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    return parser.parse_args()

def format_examples(problems, solutions):
    return [
        f"Problem: {problem}\nSolution: {solution}"
        for problem, solution in zip(problems, solutions)
    ]

def main():
    args = parse_args()

    if not os.path.exists(args.dataset_path):
        logger.error(f"Dataset not found: {args.dataset_path}")
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    logger.info(f"Loading dataset from {args.dataset_path}...")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    
    logger.info(f"Loading tokenizer for {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            format_examples(examples["problem"], examples["solution"]),
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    use_cuda = torch.cuda.is_available()
    
    bnb_config = None
    if args.use_qlora:
        try:
            import bitsandbytes
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16 if use_cuda else torch.float32,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using QLoRA 4-bit quantization.")
        except ImportError:
            logger.warning("bitsandbytes not installed. Falling back to standard LoRA.")

    logger.info(f"Loading base model {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token=args.hf_token,
        quantization_config=bnb_config,
        device_map="auto" if use_cuda else None,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
    )
    
    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    model.config.pad_token_id = tokenizer.pad_token_id
    
    logger.info("Configuring PEFT/LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=use_cuda,
        report_to="none",
        push_to_hub=args.push_to_hub,
        hub_token=args.hf_token if args.push_to_hub else None,
        remove_unused_columns=False,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving adapter to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Training complete.")

if __name__ == "__main__":
    main()

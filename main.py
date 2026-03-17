import argparse
import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def ensure_usable_temp_dir() -> str:
    # Always create a project-local temp directory first
    local_tmp = Path.cwd() / ".tmp"
    try:
        local_tmp.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(local_tmp)
        os.environ["TMP"] = str(local_tmp)
        os.environ["TEMP"] = str(local_tmp)
        os.environ["TEMPDIR"] = str(local_tmp)
        return str(local_tmp)
    except Exception:
        pass

    candidates = []

    for env_name in ("TMPDIR", "TMP", "TEMP", "TEMPDIR"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value).expanduser())

    candidates.extend(
        [
            Path("/var/folders/2z/m4gpjz516pq2lf7mqjqgrj9w0000gn/T"),
            Path("/tmp"),
            Path("/var/tmp"),
            Path("/usr/tmp"),
            Path.cwd() / ".tmp",
        ]
    )

    seen = set()
    ordered_candidates = []
    for candidate in candidates:
        resolved = str(candidate)
        if resolved not in seen:
            seen.add(resolved)
            ordered_candidates.append(candidate)

    for candidate in ordered_candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            temp_dir = str(candidate)
            os.environ["TMPDIR"] = temp_dir
            os.environ["TMP"] = temp_dir
            os.environ["TEMP"] = temp_dir
            os.environ["TEMPDIR"] = temp_dir
            return temp_dir
        except Exception:
            continue

    raise RuntimeError(
        "No usable temporary directory found. Please create a writable temp directory and set TMPDIR."
    )


ensure_usable_temp_dir()

import torch
from datasets import Dataset, DatasetDict, load_dataset
from fastapi import FastAPI
from pydantic import BaseModel
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
import uvicorn


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("gemma3_math")
temp_dir_in_use = os.environ.get("TMPDIR", "")
logger.info("Using temporary directory: %s", temp_dir_in_use)

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful mathematics tutor. Solve problems step by step, "
    "show concise reasoning, and end every final answer with the exact form "
    "Final answer: \\boxed{...}."
)


@dataclass
class ScriptConfig:
    model_name: str = "google/gemma-3-1b-it"
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    output_dir: str = "./outputs/gemma3-1b-math"
    merged_dir: str = "./outputs/gemma3-1b-math/merged"
    max_length: int = 1024
    train_samples: Optional[int] = None
    eval_samples: int = 256
    exact_match_samples: int = 128
    learning_rate: float = 2e-4
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    seed: int = 42
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    use_4bit: bool = True
    server_host: str = "0.0.0.0"
    server_port: int = 1111
    generation_max_new_tokens: int = 256
    generation_temperature: float = 0.2
    generation_top_p: float = 0.95


class GenerationRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.95


class GenerationResponse(BaseModel):
    response: str
    extracted_answer: Optional[str] = None


class SupervisedDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ----------------------------
# Utility helpers
# ----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detect_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def extract_gsm8k_reasoning_and_answer(answer_text: str) -> Tuple[str, str]:
    answer_text = answer_text.strip()
    if "####" in answer_text:
        reasoning, final = answer_text.rsplit("####", 1)
        return reasoning.strip(), final.strip()
    return answer_text, answer_text.splitlines()[-1].strip()


def normalize_math_answer(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()

    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        text = boxed[-1].strip()
    elif "####" in text:
        text = text.rsplit("####", 1)[-1].strip()
    else:
        match = re.search(r"Final answer\s*:\s*(.*)$", text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            text = match.group(1).strip()

    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.strip().strip(".")
    text = re.sub(r"\s+", "", text)
    return text


# ----------------------------
# Dataset loading + parsing
# ----------------------------
def load_math_dataset(cfg: ScriptConfig) -> DatasetDict:
    logger.info("Loading dataset %s (%s)", cfg.dataset_name, cfg.dataset_config)
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config)

    if "train" not in ds:
        raise ValueError("Dataset must provide a train split.")

    if "test" not in ds and "validation" not in ds:
        split = ds["train"].train_test_split(test_size=0.02, seed=cfg.seed)
        ds = DatasetDict(train=split["train"], test=split["test"])

    eval_split_name = "test" if "test" in ds else "validation"

    train_ds = ds["train"]
    eval_ds = ds[eval_split_name]

    if cfg.train_samples:
        train_ds = train_ds.select(range(min(cfg.train_samples, len(train_ds))))
    if cfg.eval_samples:
        eval_ds = eval_ds.select(range(min(cfg.eval_samples, len(eval_ds))))

    logger.info("Loaded %d train rows and %d eval rows", len(train_ds), len(eval_ds))
    return DatasetDict(train=train_ds, eval=eval_ds)


def parse_gsm8k_row(row: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
    question = row["question"].strip()
    reasoning, final_answer = extract_gsm8k_reasoning_and_answer(row["answer"])
    assistant = (
        f"{reasoning.strip()}\n\n"
        f"Final answer: \\boxed{{{final_answer}}}"
    ).strip()
    return {
        "system": system_prompt,
        "user": question,
        "assistant": assistant,
        "target_answer": final_answer,
    }


def parse_dataset(ds: DatasetDict, system_prompt: str) -> DatasetDict:
    parsed_train = ds["train"].map(
        lambda row: parse_gsm8k_row(row, system_prompt),
        remove_columns=ds["train"].column_names,
        desc="Parsing train split",
    )
    parsed_eval = ds["eval"].map(
        lambda row: parse_gsm8k_row(row, system_prompt),
        remove_columns=ds["eval"].column_names,
        desc="Parsing eval split",
    )
    return DatasetDict(train=parsed_train, eval=parsed_eval)


# ----------------------------
# Tokenization
# ----------------------------
def build_chat_text(tokenizer, system_prompt: str, user_text: str, assistant_text: Optional[str] = None) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tokenize_supervised_example(
    example: Dict[str, Any],
    tokenizer,
    max_length: int,
) -> Dict[str, Any]:
    prompt_text = build_chat_text(tokenizer, example["system"], example["user"], assistant_text=None)
    full_text = build_chat_text(tokenizer, example["system"], example["user"], assistant_text=example["assistant"])

    prompt_tokens = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    full_tokens = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    prompt_len = min(len(prompt_tokens["input_ids"]), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    labels = labels[: len(input_ids)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "target_answer": example["target_answer"],
        "user": example["user"],
        "system": example["system"],
    }


def tokenize_dataset(parsed_ds: DatasetDict, tokenizer, max_length: int) -> DatasetDict:
    tokenized_train = parsed_ds["train"].map(
        lambda ex: tokenize_supervised_example(ex, tokenizer, max_length),
        remove_columns=parsed_ds["train"].column_names,
        desc="Tokenizing train split",
    )
    tokenized_eval = parsed_ds["eval"].map(
        lambda ex: tokenize_supervised_example(ex, tokenizer, max_length),
        remove_columns=parsed_ds["eval"].column_names,
        desc="Tokenizing eval split",
    )
    return DatasetDict(train=tokenized_train, eval=tokenized_eval)


# ----------------------------
# Model loading
# ----------------------------
def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(cfg: ScriptConfig, for_training: bool = True):
    dtype = detect_dtype()
    quant_config = None
    device_map = "auto" if torch.cuda.is_available() else None

    if cfg.use_4bit and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
        device_map=device_map,
        quantization_config=quant_config,
        attn_implementation="eager",
    )

    if for_training:
        model.config.use_cache = False
        if quant_config is not None:
            model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        model.config.use_cache = True

    return model


# ----------------------------
# Training / evaluation
# ----------------------------
def build_training_args(cfg: ScriptConfig) -> TrainingArguments:
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    return TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        report_to="none",
        bf16=bf16,
        fp16=fp16,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )


@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    prompt: str,
    system_prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    prompt_text = build_chat_text(tokenizer, system_prompt, prompt, assistant_text=None)
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def evaluate_exact_match(
    model,
    tokenizer,
    parsed_eval_ds: Dataset,
    cfg: ScriptConfig,
) -> Dict[str, Any]:
    total = min(cfg.exact_match_samples, len(parsed_eval_ds))
    correct = 0
    rows = []

    logger.info("Running generation-based exact match on %d samples", total)
    for i in range(total):
        row = parsed_eval_ds[i]
        pred = generate_answer(
            model,
            tokenizer,
            prompt=row["user"],
            system_prompt=row["system"],
            max_new_tokens=cfg.generation_max_new_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        gold = row["target_answer"]
        pred_norm = normalize_math_answer(pred)
        gold_norm = normalize_math_answer(gold)
        is_correct = pred_norm == gold_norm
        correct += int(is_correct)
        rows.append(
            {
                "question": row["user"],
                "prediction": pred,
                "prediction_normalized": pred_norm,
                "gold": gold,
                "gold_normalized": gold_norm,
                "correct": is_correct,
            }
        )

        if (i + 1) % 10 == 0:
            logger.info("Evaluated %d/%d exact-match samples", i + 1, total)

    accuracy = correct / total if total else 0.0
    report = {
        "exact_match": accuracy,
        "num_samples": total,
        "num_correct": correct,
        "examples": rows[:10],
    }
    return report


def maybe_merge_adapter(cfg: ScriptConfig) -> str:
    logger.info("Merging LoRA adapter into base model for standalone inference")
    dtype = detect_dtype()
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    merged = PeftModel.from_pretrained(base, cfg.output_dir)
    merged = merged.merge_and_unload()
    os.makedirs(cfg.merged_dir, exist_ok=True)
    merged.save_pretrained(cfg.merged_dir)
    tokenizer = load_tokenizer(cfg.model_name)
    tokenizer.save_pretrained(cfg.merged_dir)
    logger.info("Merged model saved to %s", cfg.merged_dir)
    return cfg.merged_dir


def run_train(cfg: ScriptConfig) -> None:
    set_seed(cfg.seed)
    tokenizer = load_tokenizer(cfg.model_name)

    raw_ds = load_math_dataset(cfg)
    parsed_ds = parse_dataset(raw_ds, DEFAULT_SYSTEM_PROMPT)
    tokenized_ds = tokenize_dataset(parsed_ds, tokenizer, cfg.max_length)

    model = load_base_model(cfg, for_training=True)
    collator = SupervisedDataCollator(tokenizer)
    training_args = build_training_args(cfg)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["eval"],
        tokenizer=tokenizer,
        data_collator=collator,
    )

    logger.info("Starting training")
    train_result = trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    metrics = train_result.metrics
    metrics["train_samples"] = len(tokenized_ds["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("Running loss-based evaluation")
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(tokenized_ds["eval"])
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info("Running generation-based exact match")
    exact_match_report = evaluate_exact_match(model, tokenizer, parsed_ds["eval"], cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "exact_match_report.json"), "w", encoding="utf-8") as f:
        json.dump(exact_match_report, f, indent=2, ensure_ascii=False)
    logger.info("Exact match: %.4f (%d/%d)", exact_match_report["exact_match"], exact_match_report["num_correct"], exact_match_report["num_samples"])

    maybe_merge_adapter(cfg)
    logger.info("Training complete")


def load_inference_model(cfg: ScriptConfig):
    merged_path = cfg.merged_dir if os.path.isdir(cfg.merged_dir) else None
    source = merged_path or cfg.output_dir

    if merged_path:
        logger.info("Loading merged model from %s", merged_path)
        tokenizer = AutoTokenizer.from_pretrained(merged_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            merged_path,
            torch_dtype=detect_dtype(),
            device_map="auto" if torch.cuda.is_available() else None,
        )
        return model, tokenizer

    logger.info("Merged model not found. Loading base model + adapter from %s", source)
    tokenizer = load_tokenizer(cfg.model_name)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=detect_dtype(),
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, cfg.output_dir)
    return model, tokenizer


def run_eval(cfg: ScriptConfig) -> None:
    set_seed(cfg.seed)
    raw_ds = load_math_dataset(cfg)
    parsed_ds = parse_dataset(raw_ds, DEFAULT_SYSTEM_PROMPT)
    model, tokenizer = load_inference_model(cfg)
    report = evaluate_exact_match(model, tokenizer, parsed_ds["eval"], cfg)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def run_infer(cfg: ScriptConfig, prompt: str) -> None:
    model, tokenizer = load_inference_model(cfg)
    answer = generate_answer(
        model,
        tokenizer,
        prompt=prompt,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        max_new_tokens=cfg.generation_max_new_tokens,
        temperature=cfg.generation_temperature,
        top_p=cfg.generation_top_p,
    )
    print(answer)


def run_server(cfg: ScriptConfig) -> None:
    model, tokenizer = load_inference_model(cfg)
    app = FastAPI(title="Gemma3 1B Math Server")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "model": cfg.model_name, "port": cfg.server_port}

    @app.post("/generate", response_model=GenerationResponse)
    def generate(req: GenerationRequest) -> GenerationResponse:
        system_prompt = req.system_prompt or DEFAULT_SYSTEM_PROMPT
        response = generate_answer(
            model,
            tokenizer,
            prompt=req.prompt,
            system_prompt=system_prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        return GenerationResponse(
            response=response,
            extracted_answer=normalize_math_answer(response) or None,
        )

    uvicorn.run(app, host=cfg.server_host, port=cfg.server_port)


# ----------------------------
# CLI
# ----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 3 1B for math and serve it on port 1111.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model-name", default="google/gemma-3-1b-it")
        p.add_argument("--dataset-name", default="openai/gsm8k")
        p.add_argument("--dataset-config", default="main")
        p.add_argument("--output-dir", default="./outputs/gemma3-1b-math")
        p.add_argument("--merged-dir", default="./outputs/gemma3-1b-math/merged")
        p.add_argument("--max-length", type=int, default=1024)
        p.add_argument("--train-samples", type=int, default=None)
        p.add_argument("--eval-samples", type=int, default=256)
        p.add_argument("--exact-match-samples", type=int, default=128)
        p.add_argument("--learning-rate", type=float, default=2e-4)
        p.add_argument("--num-train-epochs", type=float, default=2.0)
        p.add_argument("--per-device-train-batch-size", type=int, default=2)
        p.add_argument("--per-device-eval-batch-size", type=int, default=2)
        p.add_argument("--gradient-accumulation-steps", type=int, default=8)
        p.add_argument("--warmup-ratio", type=float, default=0.03)
        p.add_argument("--logging-steps", type=int, default=10)
        p.add_argument("--eval-steps", type=int, default=200)
        p.add_argument("--save-steps", type=int, default=200)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--lora-r", type=int, default=32)
        p.add_argument("--lora-alpha", type=int, default=64)
        p.add_argument("--lora-dropout", type=float, default=0.05)
        p.add_argument("--use-4bit", action="store_true")
        p.add_argument("--no-use-4bit", dest="use_4bit", action="store_false")
        p.set_defaults(use_4bit=True)
        p.add_argument("--generation-max-new-tokens", type=int, default=256)
        p.add_argument("--generation-temperature", type=float, default=0.2)
        p.add_argument("--generation-top-p", type=float, default=0.95)

    train_p = subparsers.add_parser("train", help="Fine-tune the model.")
    add_common_args(train_p)

    eval_p = subparsers.add_parser("eval", help="Run exact-match evaluation with generation.")
    add_common_args(eval_p)

    infer_p = subparsers.add_parser("infer", help="Generate one answer from the command line.")
    add_common_args(infer_p)
    infer_p.add_argument("--prompt", required=True)

    serve_p = subparsers.add_parser("serve", help="Start a FastAPI server on port 1111.")
    add_common_args(serve_p)
    serve_p.add_argument("--server-host", default="0.0.0.0")
    serve_p.add_argument("--server-port", type=int, default=1111)

    return parser


def args_to_config(args: argparse.Namespace) -> ScriptConfig:
    return ScriptConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        output_dir=args.output_dir,
        merged_dir=args.merged_dir,
        max_length=args.max_length,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        exact_match_samples=args.exact_match_samples,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_4bit=args.use_4bit,
        server_host=getattr(args, "server_host", "0.0.0.0"),
        server_port=getattr(args, "server_port", 1111),
        generation_max_new_tokens=args.generation_max_new_tokens,
        generation_temperature=args.generation_temperature,
        generation_top_p=args.generation_top_p,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = args_to_config(args)

    if args.command == "train":
        run_train(cfg)
    elif args.command == "eval":
        run_eval(cfg)
    elif args.command == "infer":
        run_infer(cfg, args.prompt)
    elif args.command == "serve":
        run_server(cfg)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

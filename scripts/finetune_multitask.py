"""
Multi-task fine-tune TinyLLaMA for trend detection + causal reasoning.

Training config:
- Model: TinyLLaMA 1.1B
- Method: LoRA (r=16, alpha=32)
- Tasks: Trend detection + Causal reasoning
- Epochs: 5
- Batch size: 4
- Learning rate: 2e-4
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

PROJECT_ROOT = Path(__file__).parent.parent


def load_training_data(data_path: Path) -> List[Dict]:
    """Load training data from JSON."""
    with open(data_path) as f:
        data = json.load(f)
    return data


def format_sample(sample: Dict) -> str:
    """Format sample for training."""
    prompt = sample["prompt"]
    completion = sample["completion"]
    return f"{prompt}\n\nAnswer: {completion}"


def prepare_dataset(data: List[Dict], tokenizer) -> Dataset:
    """Prepare dataset for training."""

    # Format texts
    texts = [format_sample(sample) for sample in data]

    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None
    )

    # Create HuggingFace Dataset
    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    })

    return dataset


def main(
    train_path: Path,
    val_path: Path,
    output_dir: Path,
    n_epochs: int = 5
):
    """Multi-task fine-tune TinyLLaMA."""

    print("=" * 80)
    print("MULTI-TASK FINE-TUNING: Trend + Causal")
    print("=" * 80)
    print()

    # Load data
    print("[1/6] Loading multi-task training data...")
    train_data = load_training_data(train_path)
    val_data = load_training_data(val_path)

    # Count tasks
    trend_train = sum(1 for x in train_data if x.get("task_type") == "trend")
    causal_train = sum(1 for x in train_data if x.get("task_type") == "causal")

    print(f"  Train: {len(train_data)} samples (trend: {trend_train}, causal: {causal_train})")
    print(f"  Val:   {len(val_data)} samples")

    # Load tokenizer
    print("\n[2/6] Loading tokenizer...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  ✓ Loaded tokenizer for {model_name}")

    # Prepare datasets
    print("\n[3/6] Preparing datasets...")
    train_dataset = prepare_dataset(train_data, tokenizer)
    val_dataset = prepare_dataset(val_data, tokenizer)
    print(f"  ✓ Tokenized {len(train_dataset)} train + {len(val_dataset)} val samples")

    # Load model with 4-bit quantization
    print("\n[4/6] Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    print(f"  ✓ Loaded {model_name}")

    # Prepare for LoRA
    print("\n[5/6] Applying LoRA...")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"tinyllama_multitask_{timestamp}"

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=n_epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,
        report_to="none",
        remove_unused_columns=False
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    # Train
    print(f"\n[6/6] Starting training ({n_epochs} epochs)...")
    print(f"  Output: {output_path}")
    print()

    trainer.train()

    # Save final model
    print("\n" + "=" * 80)
    print("Saving multi-task fine-tuned model...")
    model.save_pretrained(output_path / "final")
    tokenizer.save_pretrained(output_path / "final")

    # Save training info
    info = {
        "model": model_name,
        "method": "LoRA",
        "lora_r": 16,
        "lora_alpha": 32,
        "epochs": n_epochs,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "task_distribution": {
            "trend": trend_train,
            "causal": causal_train
        },
        "timestamp": timestamp,
        "output_dir": str(output_path)
    }

    with open(output_path / "training_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    print("=" * 80)
    print("✓ MULTI-TASK FINE-TUNING COMPLETED!")
    print("=" * 80)
    print(f"\nModel saved to: {output_path / 'final'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-task fine-tune TinyLLaMA")
    parser.add_argument("--train", type=str, default="data/processed/multitask/train.json")
    parser.add_argument("--val", type=str, default="data/processed/multitask/val.json")
    parser.add_argument("--output", type=str, default="output/models")
    parser.add_argument("--epochs", type=int, default=5)

    args = parser.parse_args()

    train_path = PROJECT_ROOT / args.train
    val_path = PROJECT_ROOT / args.val
    output_dir = PROJECT_ROOT / args.output

    if not train_path.exists():
        print(f"ERROR: Training data not found at {train_path}")
        sys.exit(1)

    main(train_path, val_path, output_dir, n_epochs=args.epochs)

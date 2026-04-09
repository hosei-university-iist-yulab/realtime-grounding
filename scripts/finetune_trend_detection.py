"""
Fine-tune TinyLLaMA for trend detection using LoRA.

Training config:
- Model: TinyLLaMA 1.1B
- Method: LoRA (r=16, alpha=32)
- Epochs: 5
- Batch size: 4
- Learning rate: 2e-4
- GPU: CUDA device 0 (via CUDA_VISIBLE_DEVICES)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from dataclasses import dataclass

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


def format_prompt_completion(prompt: str, completion: str) -> str:
    """Format prompt and completion for training."""
    return f"{prompt}\n\nAnswer: {completion}"


def prepare_dataset(data: List[Dict], tokenizer) -> Dataset:
    """Prepare dataset for training."""

    # Format texts
    texts = [
        format_prompt_completion(sample["prompt"], sample["completion"])
        for sample in data
    ]

    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding=False,  # Will pad in collator
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
    """Fine-tune TinyLLaMA on trend detection task."""

    print("=" * 80)
    print("FINE-TUNING TinyLLaMA FOR TREND DETECTION")
    print("=" * 80)
    print()

    # Load data
    print("[1/6] Loading training data...")
    train_data = load_training_data(train_path)
    val_data = load_training_data(val_path)
    print(f"  Train: {len(train_data)} samples")
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
    output_path = output_dir / f"tinyllama_trend_{timestamp}"

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
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=False
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
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
    print("Saving fine-tuned model...")
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
        "timestamp": timestamp,
        "output_dir": str(output_path)
    }

    with open(output_path / "training_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    print("=" * 80)
    print("✓ FINE-TUNING COMPLETED!")
    print("=" * 80)
    print(f"\nModel saved to: {output_path / 'final'}")
    print("\nTo use the fine-tuned model:")
    print(f"  from peft import PeftModel")
    print(f"  base_model = AutoModelForCausalLM.from_pretrained('{model_name}')")
    print(f"  model = PeftModel.from_pretrained(base_model, '{output_path / 'final'}')")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune TinyLLaMA for trend detection")
    parser.add_argument("--train", type=str, default="data/processed/trend_detection/train.json")
    parser.add_argument("--val", type=str, default="data/processed/trend_detection/val.json")
    parser.add_argument("--output", type=str, default="output/models")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")

    args = parser.parse_args()

    train_path = PROJECT_ROOT / args.train
    val_path = PROJECT_ROOT / args.val
    output_dir = PROJECT_ROOT / args.output

    if not train_path.exists():
        print(f"ERROR: Training data not found at {train_path}")
        print("Run prepare_trend_training_data.py first!")
        sys.exit(1)

    if not val_path.exists():
        print(f"ERROR: Validation data not found at {val_path}")
        print("Run prepare_trend_training_data.py first!")
        sys.exit(1)

    main(train_path, val_path, output_dir, n_epochs=args.epochs)

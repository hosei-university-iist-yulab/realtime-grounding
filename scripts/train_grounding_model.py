#!/usr/bin/env python3
"""
Train Grounding Model with LoRA Fine-tuning.

Fine-tunes TinyLLaMA or Phi-2 on sensor-text grounding task
using LoRA for efficient adaptation.

Usage:
    python scripts/train_grounding_model.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --train_data data/training/train.jsonl \
        --output_dir output/models \
        --epochs 5
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType
    )
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install transformers peft bitsandbytes accelerate")
    sys.exit(1)


# Use GPU 4-7 as specified
ALLOWED_GPUS = [4, 5, 6, 7]


class GroundingDataset(Dataset):
    """Dataset for sensor-text grounding training."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512
    ):
        """
        Load training data.

        Args:
            data_path: Path to JSONL training file
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # Load data
        with open(data_path, "r") as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

        print(f"Loaded {len(self.examples)} training examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Format as instruction-following
        if "messages" in example:
            # Chat format
            text = self._format_chat(example["messages"])
        else:
            # Simple input/output format
            text = self._format_simple(example)

        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["input_ids"].squeeze()
        }

    def _format_chat(self, messages: List[Dict]) -> str:
        """Format chat messages for training."""
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text += f"<|system|>\n{content}</s>\n"
            elif role == "user":
                text += f"<|user|>\n{content}</s>\n"
            elif role == "assistant":
                text += f"<|assistant|>\n{content}</s>\n"
        return text

    def _format_simple(self, example: Dict) -> str:
        """Format simple input/output for training."""
        input_text = example.get("input_text", "")
        output_text = example.get("output_text", "")

        return (
            f"<|system|>\n"
            f"You are an energy monitoring assistant. Analyze sensor data and provide accurate insights.</s>\n"
            f"<|user|>\n{input_text}</s>\n"
            f"<|assistant|>\n{output_text}</s>\n"
        )


def setup_model(
    model_name: str,
    use_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    device_map: str = "auto"
):
    """
    Load and prepare model with LoRA.

    Args:
        model_name: HuggingFace model name
        use_4bit: Use 4-bit quantization
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        device_map: Device mapping strategy

    Returns:
        (model, tokenizer) tuple
    """
    print(f"Loading model: {model_name}")

    # Quantization config
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    # Prepare for LoRA
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Get target modules based on model
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    if "phi" in model_name.lower():
        target_modules = ["q_proj", "v_proj", "k_proj", "dense"]

    # LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train(
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    gradient_accumulation_steps: int = 4,
    warmup_ratio: float = 0.1,
    save_steps: int = 500,
    logging_steps: int = 50
):
    """
    Train model using HuggingFace Trainer.

    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset
        output_dir: Output directory for checkpoints
        epochs: Number of epochs
        batch_size: Per-device batch size
        learning_rate: Learning rate
        gradient_accumulation_steps: Gradient accumulation steps
        warmup_ratio: Warmup ratio
        save_steps: Save checkpoint every N steps
        logging_steps: Log every N steps
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=save_steps if val_dataset else None,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",  # Disable wandb etc.
        remove_unused_columns=False,
        dataloader_num_workers=4
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nSaved final model to {final_path}")

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train TGP grounding model")

    # Model arguments
    parser.add_argument("--model", type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Model name or path")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization")

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")

    # Data arguments
    parser.add_argument("--train_data", type=str,
                        default="data/training/train.jsonl",
                        help="Training data path")
    parser.add_argument("--val_data", type=str, default=None,
                        help="Validation data path")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")

    # Training arguments
    parser.add_argument("--output_dir", type=str,
                        default="output/models",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--gradient_accumulation", type=int, default=4,
                        help="Gradient accumulation steps")

    # GPU arguments
    parser.add_argument("--device", type=int, default=4,
                        help="GPU device (4-7)")

    args = parser.parse_args()

    # Validate GPU
    if args.device not in ALLOWED_GPUS:
        print(f"Warning: GPU {args.device} not in allowed range {ALLOWED_GPUS}")
        args.device = 4

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    print(f"Using GPU {args.device}")

    # Check if training data exists
    if not Path(args.train_data).exists():
        print(f"Training data not found: {args.train_data}")
        print("Generate training data first:")
        print("  python scripts/generate_training_data.py")
        sys.exit(1)

    # Setup model
    model, tokenizer = setup_model(
        args.model,
        use_4bit=not args.no_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    # Load datasets
    train_dataset = GroundingDataset(
        args.train_data,
        tokenizer,
        max_length=args.max_length
    )

    val_dataset = None
    if args.val_data and Path(args.val_data).exists():
        val_dataset = GroundingDataset(
            args.val_data,
            tokenizer,
            max_length=args.max_length
        )

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"grounding_{timestamp}"

    # Train
    trainer = train(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation
    )

    print("\nTraining complete!")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()

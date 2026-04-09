"""
LLM Backbone for Temporal Grounding Pipeline.

Supports TinyLLaMA (1.1B), Phi-2 (2.7B), and Qwen-2.5 (3B) with LoRA fine-tuning.
Designed for real-time inference on building energy data.
"""

import os
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import torch
from torch import nn

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        GenerationConfig
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftModel
    )
except ImportError as e:
    raise ImportError(f"Required packages missing: {e}")


# Use GPU 4-7 as specified in project config
ALLOWED_GPUS = [4, 5, 6, 7]


@dataclass
class ModelConfig:
    """Configuration for LLM backbone."""
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    use_4bit: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    device_map: str = "auto"
    gpu_id: Optional[int] = None  # Specific GPU to use (4-7)


# Predefined model configurations
MODEL_CONFIGS = {
    "tinyllama": ModelConfig(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_length=512
    ),
    "phi2": ModelConfig(
        model_name="microsoft/phi-2",
        max_length=1024
    ),
    "phi3-mini": ModelConfig(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        max_length=1024
    ),
    "qwen": ModelConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_length=2048
    )
}


class LLMBackbone:
    """
    LLM backbone for temporal grounding tasks.

    Supports:
    - TinyLLaMA 1.1B (fast, lower memory)
    - Phi-2 2.7B (better quality, more memory)
    - Phi-3 Mini 3.8B (best quality, most memory)
    - Qwen-2.5 3B (good quality, 2048 context)

    Features:
    - 4-bit quantization for memory efficiency
    - LoRA for efficient fine-tuning
    - Batched inference support
    - GPU assignment (devices 4-7 only)
    """

    def __init__(self, config: Optional[ModelConfig] = None, model_type: str = "tinyllama"):
        """
        Initialize LLM backbone.

        Args:
            config: Model configuration (overrides model_type if provided)
            model_type: Predefined model type ("tinyllama", "phi2", "phi3-mini")
        """
        if config is None:
            if model_type not in MODEL_CONFIGS:
                raise ValueError(f"Unknown model type: {model_type}. "
                               f"Options: {list(MODEL_CONFIGS.keys())}")
            config = MODEL_CONFIGS[model_type]

        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None

        # Set GPU if specified
        if config.gpu_id is not None:
            if config.gpu_id not in ALLOWED_GPUS:
                raise ValueError(f"GPU {config.gpu_id} not allowed. Use one of {ALLOWED_GPUS}")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

        self._load_model()

    def _load_model(self):
        """Load model with quantization and LoRA if configured."""
        print(f"Loading {self.config.model_name}...")

        # Quantization config
        bnb_config = None
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map=self.config.device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

        # Prepare for LoRA if enabled
        if self.config.use_lora:
            if self.config.use_4bit:
                self.model = prepare_model_for_kbit_training(self.model)

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self._get_target_modules(),
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # Set device
        self.device = next(self.model.parameters()).device
        print(f"Model loaded on {self.device}")

    def _get_target_modules(self) -> List[str]:
        """Get LoRA target modules based on model architecture."""
        model_name = self.config.model_name.lower()

        if "llama" in model_name or "tinyllama" in model_name:
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif "phi" in model_name:
            return ["q_proj", "v_proj", "k_proj", "dense"]
        elif "qwen" in model_name:
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        else:
            # Default targets
            return ["q_proj", "v_proj"]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (uses config default if None)
            top_p: Top-p sampling (uses config default if None)
            do_sample: Whether to use sampling

        Returns:
            Generated text
        """
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length - max_new_tokens
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode only new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate per prompt
            **kwargs: Additional generation arguments

        Returns:
            List of generated texts
        """
        # Tokenize with padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length - max_new_tokens
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=kwargs.get("temperature", self.config.temperature),
                top_p=kwargs.get("top_p", self.config.top_p),
                do_sample=kwargs.get("do_sample", True),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        responses = []
        for i, output in enumerate(outputs):
            input_len = inputs["input_ids"][i].ne(self.tokenizer.pad_token_id).sum()
            response = self.tokenizer.decode(
                output[input_len:],
                skip_special_tokens=True
            )
            responses.append(response.strip())

        return responses

    def format_grounding_prompt(
        self,
        sensor_data: Dict[str, Any],
        query: str
    ) -> str:
        """
        Format prompt for temporal grounding task.

        Args:
            sensor_data: Dict with sensor readings and metadata
            query: User query about the sensor data

        Returns:
            Formatted prompt string
        """
        # Extract key information
        building = sensor_data.get("building_id", "Unknown")
        meter = sensor_data.get("meter_type", "electricity")
        readings = sensor_data.get("readings", [])
        stats = sensor_data.get("statistics", {})

        # Build context
        context = f"Building: {building}\nMeter Type: {meter}\n"

        if stats:
            context += (
                f"Recent Statistics (last hour):\n"
                f"  - Mean: {stats.get('mean', 0):.2f} kWh\n"
                f"  - Std Dev: {stats.get('std', 0):.2f} kWh\n"
                f"  - Min: {stats.get('min', 0):.2f} kWh\n"
                f"  - Max: {stats.get('max', 0):.2f} kWh\n"
                f"  - Count: {stats.get('count', 0)} readings\n"
            )

        if readings:
            context += "Recent Readings:\n"
            for r in readings[-5:]:  # Show last 5
                context += f"  - {r.get('timestamp', 'N/A')}: {r.get('value', 0):.2f} kWh\n"

        # Format as chat
        prompt = (
            f"<|system|>\n"
            f"You are an energy monitoring assistant. Analyze sensor data and provide "
            f"accurate, real-time insights about building energy consumption.</s>\n"
            f"<|user|>\n"
            f"Context:\n{context}\n"
            f"Question: {query}</s>\n"
            f"<|assistant|>\n"
        )

        return prompt

    def save_lora(self, path: str):
        """Save LoRA weights."""
        if not self.config.use_lora:
            raise ValueError("LoRA not enabled, nothing to save")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"LoRA weights saved to {path}")

    def load_lora(self, path: str):
        """Load LoRA weights from path."""
        self.model = PeftModel.from_pretrained(self.model, path)
        print(f"LoRA weights loaded from {path}")

    def benchmark_latency(self, n_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark inference latency.

        Args:
            n_runs: Number of runs to average

        Returns:
            Dict with latency statistics
        """
        test_prompt = self.format_grounding_prompt(
            {"building_id": "test", "meter_type": "electricity"},
            "What is the current energy consumption?"
        )

        latencies = []

        # Warmup
        _ = self.generate(test_prompt, max_new_tokens=50)

        # Benchmark
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = self.generate(test_prompt, max_new_tokens=50)
            latencies.append((time.perf_counter() - start) * 1000)

        return {
            "mean_ms": sum(latencies) / len(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "n_runs": n_runs
        }

    def get_memory_usage(self) -> Dict[str, float]:
        """Get GPU memory usage in MB."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        allocated = torch.cuda.memory_allocated(self.device) / 1024**2
        reserved = torch.cuda.memory_reserved(self.device) / 1024**2

        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved
        }


if __name__ == "__main__":
    # Quick test
    print("Testing LLM Backbone...")

    # Use TinyLLaMA for testing (smaller, faster)
    config = ModelConfig(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        use_4bit=True,
        use_lora=True,
        gpu_id=4  # Use GPU 4
    )

    llm = LLMBackbone(config=config)

    # Test generation
    prompt = llm.format_grounding_prompt(
        {
            "building_id": "Panther_office_Leigh",
            "meter_type": "electricity",
            "statistics": {"mean": 150.5, "std": 12.3, "min": 120.0, "max": 180.0, "count": 60}
        },
        "Is the current energy consumption normal?"
    )

    print(f"\nPrompt:\n{prompt}")

    response = llm.generate(prompt, max_new_tokens=100)
    print(f"\nResponse:\n{response}")

    # Benchmark
    print("\nRunning latency benchmark...")
    latency = llm.benchmark_latency(n_runs=5)
    print(f"Latency: {latency['mean_ms']:.1f} ms (avg over {latency['n_runs']} runs)")

    # Memory
    memory = llm.get_memory_usage()
    print(f"GPU Memory: {memory['allocated_mb']:.1f} MB allocated")

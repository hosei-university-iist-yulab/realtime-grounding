"""
Prompt Engineering Baselines for Trend Detection.

Implements various prompting strategies to compare against fine-tuning:
1. Zero-shot: Direct question without examples
2. Zero-shot CoT: "Let's think step by step" reasoning
3. Few-shot: Include 3 examples in prompt
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class TrendExample:
    """Example for few-shot prompting."""
    mean: float
    std: float
    slope: float
    direction: str
    trend: str  # increasing, decreasing, stable, volatile


# Few-shot examples for trend detection
FEW_SHOT_EXAMPLES = [
    TrendExample(
        mean=120.5, std=8.2, slope=2.5,
        direction="up", trend="increasing"
    ),
    TrendExample(
        mean=145.0, std=12.1, slope=-3.2,
        direction="down", trend="decreasing"
    ),
    TrendExample(
        mean=100.0, std=5.5, slope=0.1,
        direction="stable", trend="stable"
    ),
]


class PromptFormatter:
    """
    Formats prompts with different strategies.

    Strategies:
    - zero_shot: Direct question
    - zero_shot_cot: Chain of thought reasoning
    - few_shot: Include examples
    """

    @staticmethod
    def format_zero_shot(stats: Dict, trend_info: Dict) -> str:
        """Format zero-shot prompt (baseline)."""
        return f"""You are analyzing energy consumption data for a building.

Recent statistics (last 100 readings):
- Mean: {stats['mean']:.1f} kWh
- Std: {stats['std']:.1f} kWh
- Min: {stats.get('min', 0):.1f} kWh
- Max: {stats.get('max', 0):.1f} kWh

Enhanced trend analysis:
- Direction: {trend_info.get('direction', 'unknown')}
- Slope: {trend_info.get('slope_per_hour', 0):.2f} kWh/hour
- Confidence: {trend_info.get('confidence', 0):.2f}
- R²: {trend_info.get('r_squared', 0):.3f}
- Volatility: {trend_info.get('volatility', 0):.2f}

Question: What is the trend in energy consumption?
Answer ONLY with one word: "increasing", "decreasing", "stable", or "volatile".

Answer:"""

    @staticmethod
    def format_zero_shot_cot(stats: Dict, trend_info: Dict) -> str:
        """Format zero-shot Chain of Thought prompt."""
        return f"""You are analyzing energy consumption data for a building.

Recent statistics (last 100 readings):
- Mean: {stats['mean']:.1f} kWh
- Std: {stats['std']:.1f} kWh
- Min: {stats.get('min', 0):.1f} kWh
- Max: {stats.get('max', 0):.1f} kWh

Enhanced trend analysis:
- Direction: {trend_info.get('direction', 'unknown')}
- Slope: {trend_info.get('slope_per_hour', 0):.2f} kWh/hour
- Confidence: {trend_info.get('confidence', 0):.2f}
- R²: {trend_info.get('r_squared', 0):.3f}
- Volatility: {trend_info.get('volatility', 0):.2f}

Question: What is the trend in energy consumption?

Let's think step by step:
1. First, look at the slope. A positive slope means increasing, negative means decreasing.
2. Then check confidence and R². High values mean the trend is reliable.
3. Check volatility. High volatility suggests unstable patterns.
4. Consider the direction indicator.

Based on this analysis, the trend is:"""

    @staticmethod
    def format_few_shot(stats: Dict, trend_info: Dict) -> str:
        """Format few-shot prompt with examples."""
        examples = ""
        for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
            examples += f"""
Example {i}:
- Mean: {ex.mean} kWh, Std: {ex.std} kWh
- Slope: {ex.slope} kWh/hour, Direction: {ex.direction}
Answer: {ex.trend}
"""

        return f"""You are analyzing energy consumption data for a building.

Here are some examples:
{examples}
Now analyze this data:

Recent statistics (last 100 readings):
- Mean: {stats['mean']:.1f} kWh
- Std: {stats['std']:.1f} kWh
- Min: {stats.get('min', 0):.1f} kWh
- Max: {stats.get('max', 0):.1f} kWh

Enhanced trend analysis:
- Direction: {trend_info.get('direction', 'unknown')}
- Slope: {trend_info.get('slope_per_hour', 0):.2f} kWh/hour
- Confidence: {trend_info.get('confidence', 0):.2f}
- R²: {trend_info.get('r_squared', 0):.3f}
- Volatility: {trend_info.get('volatility', 0):.2f}

Question: What is the trend in energy consumption?
Answer ONLY with one word: "increasing", "decreasing", "stable", or "volatile".

Answer:"""

    @staticmethod
    def format_causal_zero_shot(context: str, query: str) -> str:
        """Format zero-shot prompt for causal reasoning."""
        return f"""You are analyzing building energy data.

Context: {context}

Question: {query}

Provide a causal explanation for the energy consumption.

Answer:"""

    @staticmethod
    def format_causal_cot(context: str, query: str) -> str:
        """Format Chain of Thought prompt for causal reasoning."""
        return f"""You are analyzing building energy data.

Context: {context}

Question: {query}

Let's think step by step about the causal relationships:
1. What environmental factors affect energy consumption?
2. How does occupancy influence equipment and lighting usage?
3. What is the direction of causation (cause → effect)?

Based on this causal analysis:"""


def extract_trend_from_response(response: str) -> str:
    """
    Extract trend label from LLM response.

    Args:
        response: LLM response text

    Returns:
        One of: "increasing", "decreasing", "stable", "volatile", "unknown"
    """
    response_lower = response.lower()

    # Check for each trend keyword
    trends = ["increasing", "decreasing", "stable", "volatile"]
    for trend in trends:
        if trend in response_lower:
            return trend

    # Check for synonyms
    if any(w in response_lower for w in ["rising", "going up", "growing"]):
        return "increasing"
    if any(w in response_lower for w in ["falling", "going down", "dropping"]):
        return "decreasing"
    if any(w in response_lower for w in ["constant", "flat", "steady"]):
        return "stable"
    if any(w in response_lower for w in ["fluctuating", "unstable", "erratic"]):
        return "volatile"

    return "unknown"


class PromptBaseline:
    """
    Prompt engineering baseline for comparison experiments.

    Wraps an LLM model with different prompting strategies.
    """

    def __init__(self, model, tokenizer, strategy: str = "zero_shot"):
        """
        Initialize prompt baseline.

        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            strategy: One of "zero_shot", "zero_shot_cot", "few_shot"
        """
        self.model = model
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.formatter = PromptFormatter()

    def predict_trend(
        self,
        stats: Dict,
        trend_info: Dict,
        max_new_tokens: int = 50
    ) -> str:
        """
        Predict trend using the configured prompting strategy.

        Args:
            stats: Statistics dict (mean, std, min, max)
            trend_info: Trend analysis dict (direction, slope, etc.)
            max_new_tokens: Max tokens to generate

        Returns:
            Predicted trend: "increasing", "decreasing", "stable", "volatile"
        """
        import torch

        # Format prompt based on strategy
        if self.strategy == "zero_shot":
            prompt = self.formatter.format_zero_shot(stats, trend_info)
        elif self.strategy == "zero_shot_cot":
            prompt = self.formatter.format_zero_shot_cot(stats, trend_info)
        elif self.strategy == "few_shot":
            prompt = self.formatter.format_few_shot(stats, trend_info)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return extract_trend_from_response(response)

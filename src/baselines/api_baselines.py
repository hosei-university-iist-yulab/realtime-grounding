"""
API-based Baselines for Temporal Grounding Comparison.

Implements Claude and GPT-4 baselines for comparison with TGP.
Used to establish SOTA baseline performance.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class BaselineResult:
    """Result from baseline model."""
    response: str
    latency_ms: float
    model: str
    tokens_used: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None


class BaselineModel(ABC):
    """Abstract base class for baseline models."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        sensor_context: Dict[str, Any],
        max_tokens: int = 256
    ) -> BaselineResult:
        """Generate response for sensor grounding task."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass


class ClaudeBaseline(BaselineModel):
    """
    Claude API baseline for temporal grounding.

    Uses Claude 3.5 Sonnet for comparison.
    """

    # Pricing per 1M tokens (as of 2024)
    INPUT_COST_PER_1M = 3.0   # $3 per 1M input tokens
    OUTPUT_COST_PER_1M = 15.0  # $15 per 1M output tokens

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None
    ):
        """
        Initialize Claude baseline.

        Args:
            model: Claude model to use
            api_key: Anthropic API key (uses env var if not provided)
        """
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it in .env or pass directly."
            )

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

    def _format_system_prompt(self) -> str:
        """Format system prompt for grounding task."""
        return (
            "You are an expert energy monitoring assistant. Your role is to analyze "
            "real-time sensor data from buildings and provide accurate, actionable insights "
            "about energy consumption patterns. Always ground your responses in the provided "
            "sensor data and statistics. Be concise and precise."
        )

    def _format_user_prompt(
        self,
        query: str,
        sensor_context: Dict[str, Any]
    ) -> str:
        """Format user prompt with sensor context."""
        building = sensor_context.get("building_id", "Unknown")
        meter = sensor_context.get("meter_type", "electricity")
        stats = sensor_context.get("statistics", {})
        readings = sensor_context.get("readings", [])

        context = f"Building: {building}\nMeter: {meter}\n\n"

        if stats:
            context += "Statistics (last hour):\n"
            context += f"  - Mean: {stats.get('mean', 0):.2f} kWh\n"
            context += f"  - Std Dev: {stats.get('std', 0):.2f} kWh\n"
            context += f"  - Min: {stats.get('min', 0):.2f} kWh\n"
            context += f"  - Max: {stats.get('max', 0):.2f} kWh\n"
            context += f"  - Count: {stats.get('count', 0)} readings\n\n"

        if readings:
            context += "Recent readings:\n"
            for r in readings[-5:]:
                context += f"  - {r.get('timestamp', 'N/A')}: {r.get('value', 0):.2f} kWh\n"
            context += "\n"

        return f"{context}Question: {query}"

    def generate(
        self,
        prompt: str,
        sensor_context: Dict[str, Any],
        max_tokens: int = 256
    ) -> BaselineResult:
        """Generate response using Claude API."""
        start_time = time.perf_counter()

        try:
            user_prompt = self._format_user_prompt(prompt, sensor_context)

            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=self._format_system_prompt(),
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            latency = (time.perf_counter() - start_time) * 1000
            response = message.content[0].text

            # Calculate tokens and cost
            input_tokens = message.usage.input_tokens
            output_tokens = message.usage.output_tokens
            total_tokens = input_tokens + output_tokens

            cost = (
                (input_tokens / 1_000_000) * self.INPUT_COST_PER_1M +
                (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_1M
            )

            return BaselineResult(
                response=response,
                latency_ms=latency,
                model=self.model,
                tokens_used=total_tokens,
                cost_usd=cost
            )

        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            return BaselineResult(
                response="",
                latency_ms=latency,
                model=self.model,
                error=str(e)
            )

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "provider": "Anthropic",
            "model": self.model,
            "type": "api",
            "input_cost_per_1m": self.INPUT_COST_PER_1M,
            "output_cost_per_1m": self.OUTPUT_COST_PER_1M
        }


class GPT4Baseline(BaselineModel):
    """
    GPT-4 API baseline for temporal grounding.

    Uses GPT-4 Turbo for comparison.
    """

    # Pricing per 1M tokens (as of 2024)
    INPUT_COST_PER_1M = 10.0   # $10 per 1M input tokens
    OUTPUT_COST_PER_1M = 30.0  # $30 per 1M output tokens

    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        api_key: Optional[str] = None
    ):
        """
        Initialize GPT-4 baseline.

        Args:
            model: OpenAI model to use
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            print("Warning: OPENAI_API_KEY not found. GPT-4 baseline will not work.")
            self.client = None
        else:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package required: pip install openai")

    def _format_messages(
        self,
        query: str,
        sensor_context: Dict[str, Any]
    ) -> List[Dict]:
        """Format messages for GPT-4."""
        building = sensor_context.get("building_id", "Unknown")
        meter = sensor_context.get("meter_type", "electricity")
        stats = sensor_context.get("statistics", {})

        context = f"Building: {building}, Meter: {meter}\n"
        if stats:
            context += f"Stats: mean={stats.get('mean', 0):.1f}, "
            context += f"std={stats.get('std', 0):.1f}, "
            context += f"min={stats.get('min', 0):.1f}, "
            context += f"max={stats.get('max', 0):.1f} kWh"

        return [
            {
                "role": "system",
                "content": (
                    "You are an expert energy monitoring assistant. Analyze sensor data "
                    "and provide accurate insights about building energy consumption."
                )
            },
            {
                "role": "user",
                "content": f"{context}\n\nQuestion: {query}"
            }
        ]

    def generate(
        self,
        prompt: str,
        sensor_context: Dict[str, Any],
        max_tokens: int = 256
    ) -> BaselineResult:
        """Generate response using GPT-4 API."""
        if self.client is None:
            return BaselineResult(
                response="",
                latency_ms=0,
                model=self.model,
                error="OpenAI API key not configured"
            )

        start_time = time.perf_counter()

        try:
            messages = self._format_messages(prompt, sensor_context)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens
            )

            latency = (time.perf_counter() - start_time) * 1000
            text = response.choices[0].message.content

            # Calculate tokens and cost
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = input_tokens + output_tokens

            cost = (
                (input_tokens / 1_000_000) * self.INPUT_COST_PER_1M +
                (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_1M
            )

            return BaselineResult(
                response=text,
                latency_ms=latency,
                model=self.model,
                tokens_used=total_tokens,
                cost_usd=cost
            )

        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            return BaselineResult(
                response="",
                latency_ms=latency,
                model=self.model,
                error=str(e)
            )

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "provider": "OpenAI",
            "model": self.model,
            "type": "api",
            "input_cost_per_1m": self.INPUT_COST_PER_1M,
            "output_cost_per_1m": self.OUTPUT_COST_PER_1M
        }


class StaticPromptBaseline(BaselineModel):
    """
    Static prompt baseline (no real-time grounding).

    Uses fixed prompts without sensor data for comparison.
    Demonstrates the value of real-time grounding.
    """

    def __init__(self, use_claude: bool = True):
        """
        Initialize static baseline.

        Args:
            use_claude: Use Claude API (else GPT-4)
        """
        if use_claude:
            self.backend = ClaudeBaseline()
        else:
            self.backend = GPT4Baseline()

    def generate(
        self,
        prompt: str,
        sensor_context: Dict[str, Any],
        max_tokens: int = 256
    ) -> BaselineResult:
        """Generate response WITHOUT sensor context."""
        # Only use building ID, no real-time data
        minimal_context = {
            "building_id": sensor_context.get("building_id", "a building"),
            "meter_type": sensor_context.get("meter_type", "electricity")
        }

        return self.backend.generate(prompt, minimal_context, max_tokens)

    def get_model_info(self) -> Dict[str, Any]:
        info = self.backend.get_model_info()
        info["variant"] = "static_prompt"
        return info


class RAGBaseline(BaselineModel):
    """
    RAG (Retrieval-Augmented Generation) baseline.

    Retrieves historical data but doesn't do real-time grounding.
    Uses semantic search over historical patterns.
    """

    def __init__(
        self,
        backend: Optional[BaselineModel] = None,
        history_file: Optional[str] = None
    ):
        """
        Initialize RAG baseline.

        Args:
            backend: LLM backend to use
            history_file: Path to historical patterns JSON
        """
        self.backend = backend or ClaudeBaseline()
        self.history = self._load_history(history_file)

    def _load_history(self, path: Optional[str]) -> List[Dict]:
        """Load historical patterns."""
        if path and os.path.exists(path):
            with open(path) as f:
                return json.load(f)

        # Default patterns
        return [
            {
                "pattern": "high_morning",
                "description": "Morning startup spike typical 7-9 AM",
                "typical_increase": "30-50%"
            },
            {
                "pattern": "weekend_low",
                "description": "Weekend consumption 40-60% lower than weekday",
                "typical_decrease": "40-60%"
            },
            {
                "pattern": "hvac_seasonal",
                "description": "HVAC drives 40-60% of consumption in extreme weather",
                "typical_load": "40-60%"
            }
        ]

    def _retrieve_patterns(
        self,
        query: str,
        sensor_context: Dict[str, Any]
    ) -> str:
        """Retrieve relevant historical patterns."""
        # Simple keyword matching (would use embeddings in production)
        relevant = []

        query_lower = query.lower()
        for pattern in self.history:
            if any(kw in query_lower for kw in ["morning", "startup", "spike"]):
                if pattern["pattern"] == "high_morning":
                    relevant.append(pattern["description"])
            elif any(kw in query_lower for kw in ["weekend", "saturday", "sunday"]):
                if pattern["pattern"] == "weekend_low":
                    relevant.append(pattern["description"])
            elif any(kw in query_lower for kw in ["hvac", "heating", "cooling", "weather"]):
                if pattern["pattern"] == "hvac_seasonal":
                    relevant.append(pattern["description"])

        if relevant:
            return "Historical patterns: " + "; ".join(relevant)
        return ""

    def generate(
        self,
        prompt: str,
        sensor_context: Dict[str, Any],
        max_tokens: int = 256
    ) -> BaselineResult:
        """Generate response with RAG-style retrieval."""
        # Add retrieved patterns to context
        patterns = self._retrieve_patterns(prompt, sensor_context)
        if patterns:
            enhanced_context = dict(sensor_context)
            enhanced_context["historical_patterns"] = patterns
        else:
            enhanced_context = sensor_context

        return self.backend.generate(prompt, enhanced_context, max_tokens)

    def get_model_info(self) -> Dict[str, Any]:
        info = self.backend.get_model_info()
        info["variant"] = "rag"
        info["n_patterns"] = len(self.history)
        return info


def compare_baselines(
    query: str,
    sensor_context: Dict[str, Any],
    include_gpt4: bool = False
) -> Dict[str, BaselineResult]:
    """
    Compare all baselines on a single query.

    Args:
        query: Query to test
        sensor_context: Sensor data context
        include_gpt4: Include GPT-4 baseline

    Returns:
        Dict mapping baseline name to result
    """
    results = {}

    # Claude baseline
    try:
        claude = ClaudeBaseline()
        results["claude"] = claude.generate(query, sensor_context)
    except Exception as e:
        print(f"Claude baseline failed: {e}")

    # GPT-4 baseline
    if include_gpt4:
        try:
            gpt4 = GPT4Baseline()
            results["gpt4"] = gpt4.generate(query, sensor_context)
        except Exception as e:
            print(f"GPT-4 baseline failed: {e}")

    # Static baseline
    try:
        static = StaticPromptBaseline()
        results["static"] = static.generate(query, sensor_context)
    except Exception as e:
        print(f"Static baseline failed: {e}")

    # RAG baseline
    try:
        rag = RAGBaseline()
        results["rag"] = rag.generate(query, sensor_context)
    except Exception as e:
        print(f"RAG baseline failed: {e}")

    return results


if __name__ == "__main__":
    # Test baselines
    print("Testing API Baselines...")

    test_context = {
        "building_id": "Panther_office_Leigh",
        "meter_type": "electricity",
        "statistics": {
            "mean": 150.0,
            "std": 15.0,
            "min": 120.0,
            "max": 180.0,
            "count": 60
        },
        "readings": [
            {"timestamp": "2024-01-15 10:00", "value": 155.0},
            {"timestamp": "2024-01-15 10:01", "value": 152.0}
        ]
    }

    test_query = "Is the current energy consumption normal for this building?"

    print("\n--- Testing Claude Baseline ---")
    try:
        claude = ClaudeBaseline()
        result = claude.generate(test_query, test_context)
        print(f"Response: {result.response[:200]}...")
        print(f"Latency: {result.latency_ms:.0f} ms")
        print(f"Tokens: {result.tokens_used}")
        print(f"Cost: ${result.cost_usd:.6f}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Testing Static Baseline ---")
    try:
        static = StaticPromptBaseline()
        result = static.generate(test_query, test_context)
        print(f"Response: {result.response[:200]}...")
        print(f"Latency: {result.latency_ms:.0f} ms")
    except Exception as e:
        print(f"Error: {e}")

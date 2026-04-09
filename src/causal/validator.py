"""
Causal Validator for Temporal Grounding Pipeline.

Validates LLM responses against learned causal graphs to ensure
physically plausible explanations.

Integrates with Topic 1 causal discovery outputs.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None
    print("Warning: networkx not installed. Some features will be limited.")


@dataclass
class CausalEdge:
    """Represents a causal relationship."""
    cause: str
    effect: str
    strength: float = 1.0  # Causal strength/weight
    lag: int = 0           # Time lag in steps
    confidence: float = 1.0


@dataclass
class ValidationResult:
    """Result of causal validation."""
    is_valid: bool
    violations: List[str]
    score: float  # 0-1, higher = more valid
    supported_claims: List[str]
    unsupported_claims: List[str]


class CausalGraph:
    """
    Causal graph for energy consumption relationships.

    Represents causal relationships between:
    - Temperature -> Consumption
    - Occupancy -> Consumption
    - Time of day -> Consumption
    - Building type -> Base load
    - etc.
    """

    def __init__(self):
        """Initialize empty causal graph."""
        self.edges: Dict[Tuple[str, str], CausalEdge] = {}
        self.nodes: Set[str] = set()

        if nx is not None:
            self._graph = nx.DiGraph()
        else:
            self._graph = None

    def add_edge(
        self,
        cause: str,
        effect: str,
        strength: float = 1.0,
        lag: int = 0,
        confidence: float = 1.0
    ):
        """Add a causal edge."""
        edge = CausalEdge(cause, effect, strength, lag, confidence)
        self.edges[(cause, effect)] = edge
        self.nodes.add(cause)
        self.nodes.add(effect)

        if self._graph is not None:
            self._graph.add_edge(cause, effect, weight=strength, lag=lag, confidence=confidence)

    def has_edge(self, cause: str, effect: str) -> bool:
        """Check if causal edge exists."""
        return (cause, effect) in self.edges

    def get_causes(self, effect: str) -> List[str]:
        """Get all causes of an effect."""
        return [c for c, e in self.edges.keys() if e == effect]

    def get_effects(self, cause: str) -> List[str]:
        """Get all effects of a cause."""
        return [e for c, e in self.edges.keys() if c == cause]

    def get_ancestors(self, node: str) -> Set[str]:
        """Get all causal ancestors of a node."""
        if self._graph is not None:
            return set(nx.ancestors(self._graph, node))
        else:
            # Manual BFS
            ancestors = set()
            to_visit = self.get_causes(node)
            while to_visit:
                current = to_visit.pop()
                if current not in ancestors:
                    ancestors.add(current)
                    to_visit.extend(self.get_causes(current))
            return ancestors

    def get_descendants(self, node: str) -> Set[str]:
        """Get all causal descendants of a node."""
        if self._graph is not None:
            return set(nx.descendants(self._graph, node))
        else:
            descendants = set()
            to_visit = self.get_effects(node)
            while to_visit:
                current = to_visit.pop()
                if current not in descendants:
                    descendants.add(current)
                    to_visit.extend(self.get_effects(current))
            return descendants

    def is_valid_path(self, path: List[str]) -> bool:
        """Check if a causal path exists."""
        for i in range(len(path) - 1):
            if not self.has_edge(path[i], path[i + 1]):
                return False
        return True

    def save(self, path: str):
        """Save graph to JSON."""
        data = {
            "nodes": list(self.nodes),
            "edges": [
                {
                    "cause": e.cause,
                    "effect": e.effect,
                    "strength": e.strength,
                    "lag": e.lag,
                    "confidence": e.confidence
                }
                for e in self.edges.values()
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CausalGraph":
        """Load graph from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        graph = cls()
        for edge_data in data["edges"]:
            graph.add_edge(
                edge_data["cause"],
                edge_data["effect"],
                edge_data.get("strength", 1.0),
                edge_data.get("lag", 0),
                edge_data.get("confidence", 1.0)
            )
        return graph

    @classmethod
    def create_energy_graph(cls) -> "CausalGraph":
        """Create standard causal graph for building energy."""
        graph = cls()

        # Weather -> Consumption
        graph.add_edge("outdoor_temperature", "hvac_load", strength=0.8)
        graph.add_edge("humidity", "hvac_load", strength=0.3)
        graph.add_edge("solar_radiation", "lighting_load", strength=-0.4)

        # Time -> Consumption
        graph.add_edge("hour_of_day", "occupancy", strength=0.7)
        graph.add_edge("day_of_week", "occupancy", strength=0.5)
        graph.add_edge("is_holiday", "occupancy", strength=-0.8)

        # Occupancy -> Consumption
        graph.add_edge("occupancy", "plug_load", strength=0.9)
        graph.add_edge("occupancy", "lighting_load", strength=0.7)
        graph.add_edge("occupancy", "hvac_load", strength=0.4)

        # Building properties -> Base load
        graph.add_edge("floor_area", "base_load", strength=0.6)
        graph.add_edge("building_age", "efficiency", strength=-0.3)
        graph.add_edge("efficiency", "hvac_load", strength=-0.5)

        # Components -> Total
        graph.add_edge("hvac_load", "total_consumption", strength=0.5)
        graph.add_edge("lighting_load", "total_consumption", strength=0.2)
        graph.add_edge("plug_load", "total_consumption", strength=0.2)
        graph.add_edge("base_load", "total_consumption", strength=0.1)

        return graph


class CausalValidator:
    """
    Validates LLM responses against causal graphs.

    Ensures that:
    1. Claimed causes actually cause the stated effects
    2. Direction of causation is correct
    3. No physically impossible claims are made
    """

    # Keywords mapping to causal concepts
    CONCEPT_KEYWORDS = {
        "temperature": ["temperature", "temp", "hot", "cold", "warm", "cool", "heat", "thermal"],
        "outdoor_temperature": ["outdoor", "outside", "ambient", "weather"],
        "hvac_load": ["hvac", "heating", "cooling", "air conditioning", "ac"],
        "occupancy": ["occupancy", "people", "occupants", "workers", "staff", "usage"],
        "hour_of_day": ["hour", "time of day", "morning", "afternoon", "evening", "night"],
        "day_of_week": ["weekday", "weekend", "monday", "friday", "saturday", "sunday"],
        "lighting_load": ["lighting", "lights", "illumination"],
        "plug_load": ["plug", "equipment", "computers", "devices", "appliances"],
        "total_consumption": ["consumption", "energy use", "electricity", "power", "load", "demand"],
        "efficiency": ["efficiency", "efficient", "inefficient"],
        "solar_radiation": ["solar", "sun", "sunlight", "daylight"],
    }

    # Causal keywords
    CAUSAL_INDICATORS = {
        "causes": ["causes", "leads to", "results in", "produces", "creates", "drives"],
        "caused_by": ["caused by", "due to", "because of", "result of", "driven by", "from"],
        "increases": ["increases", "raises", "higher", "more", "boost"],
        "decreases": ["decreases", "reduces", "lower", "less", "drop"]
    }

    def __init__(self, causal_graph: Optional[CausalGraph] = None):
        """
        Initialize validator.

        Args:
            causal_graph: Causal graph to validate against.
                         If None, uses default energy graph.
        """
        self.graph = causal_graph or CausalGraph.create_energy_graph()

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract causal concepts from text."""
        text_lower = text.lower()
        found = []

        for concept, keywords in self.CONCEPT_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    found.append(concept)
                    break

        return found

    def _extract_causal_claims(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract causal claims from text.

        Returns list of (cause, effect, direction) tuples.
        Direction is 'positive' or 'negative'.
        """
        claims = []
        text_lower = text.lower()
        sentences = text_lower.replace(".", ". ").split(". ")

        for sentence in sentences:
            concepts = self._extract_concepts(sentence)
            if len(concepts) < 2:
                continue

            # Check for causal indicators
            has_causal = False
            direction = "positive"

            for indicator in self.CAUSAL_INDICATORS["causes"]:
                if indicator in sentence:
                    has_causal = True
                    break

            for indicator in self.CAUSAL_INDICATORS["caused_by"]:
                if indicator in sentence:
                    has_causal = True
                    concepts = concepts[::-1]  # Reverse order
                    break

            for indicator in self.CAUSAL_INDICATORS["decreases"]:
                if indicator in sentence:
                    direction = "negative"
                    break

            if has_causal and len(concepts) >= 2:
                claims.append((concepts[0], concepts[1], direction))

        return claims

    def validate(self, response: str) -> ValidationResult:
        """
        Validate an LLM response against causal graph.

        Args:
            response: LLM response text to validate

        Returns:
            ValidationResult with validity and details
        """
        claims = self._extract_causal_claims(response)
        violations = []
        supported = []
        unsupported = []

        for cause, effect, direction in claims:
            if self.graph.has_edge(cause, effect):
                # Check direction consistency
                edge = self.graph.edges[(cause, effect)]
                expected_dir = "positive" if edge.strength > 0 else "negative"

                if direction == expected_dir:
                    supported.append(f"{cause} -> {effect} ({direction})")
                else:
                    violations.append(
                        f"Direction mismatch: {cause} -> {effect} should be {expected_dir}, "
                        f"but response implies {direction}"
                    )
            elif self.graph.has_edge(effect, cause):
                # Reversed causation
                violations.append(
                    f"Reversed causation: {cause} does not cause {effect}. "
                    f"The causal direction is {effect} -> {cause}"
                )
            elif cause in self.graph.nodes and effect in self.graph.nodes:
                # Both concepts exist but no direct edge
                # Check if there's an indirect path
                descendants = self.graph.get_descendants(cause)
                if effect in descendants:
                    supported.append(f"{cause} -> ... -> {effect} (indirect)")
                else:
                    unsupported.append(
                        f"No causal path from {cause} to {effect}"
                    )
            else:
                # Unknown concepts - neutral
                pass

        # Calculate score
        total = len(supported) + len(violations) + len(unsupported)
        if total == 0:
            score = 1.0  # No claims made
        else:
            score = len(supported) / total

        is_valid = len(violations) == 0

        return ValidationResult(
            is_valid=is_valid,
            violations=violations,
            score=score,
            supported_claims=supported,
            unsupported_claims=unsupported
        )

    def suggest_correction(self, violation: str) -> str:
        """Suggest correction for a causal violation."""
        # Simple rule-based suggestions
        if "reversed causation" in violation.lower():
            return "Swap the cause and effect in the statement."
        elif "direction mismatch" in violation.lower():
            return "Change the direction of the relationship (increase/decrease)."
        elif "no causal path" in violation.lower():
            return "This relationship may be correlational, not causal."
        else:
            return "Review the causal relationship for accuracy."

    def enrich_with_causes(self, effect: str) -> List[str]:
        """Get known causes for an effect to enrich LLM context."""
        causes = self.graph.get_causes(effect)
        explanations = []

        for cause in causes:
            edge = self.graph.edges.get((cause, effect))
            if edge:
                direction = "increases" if edge.strength > 0 else "decreases"
                explanations.append(f"{cause} {direction} {effect}")

        return explanations


class Topic1Integration:
    """
    Integration with Topic 1 causal discovery outputs.

    Loads causal graphs from Topic 1 project and converts
    to format usable by CausalValidator.
    """

    TOPIC1_PATH = Path("/home/Aboya_25R9803/projects/02-SLM-Foundational/01-causal-slm")

    @classmethod
    def load_topic1_graph(cls, building_id: str = "default") -> Optional[CausalGraph]:
        """
        Load causal graph from Topic 1 outputs.

        Args:
            building_id: Building identifier for specific graph

        Returns:
            CausalGraph or None if not found
        """
        # Look for Topic 1 output files
        possible_paths = [
            cls.TOPIC1_PATH / "output" / "causal_graphs" / f"{building_id}.json",
            cls.TOPIC1_PATH / "results" / f"graph_{building_id}.json",
            cls.TOPIC1_PATH / "data" / "causal" / f"{building_id}_graph.json",
        ]

        for path in possible_paths:
            if path.exists():
                try:
                    return cls._parse_topic1_format(path)
                except Exception as e:
                    print(f"Warning: Failed to parse {path}: {e}")

        return None

    @classmethod
    def _parse_topic1_format(cls, path: Path) -> CausalGraph:
        """Parse Topic 1 causal graph format."""
        with open(path) as f:
            data = json.load(f)

        graph = CausalGraph()

        # Handle different possible formats
        if "edges" in data:
            for edge in data["edges"]:
                graph.add_edge(
                    edge.get("source", edge.get("from", edge.get("cause"))),
                    edge.get("target", edge.get("to", edge.get("effect"))),
                    edge.get("weight", edge.get("strength", 1.0)),
                    edge.get("lag", 0),
                    edge.get("confidence", 1.0)
                )
        elif "adjacency_matrix" in data:
            # Convert adjacency matrix format
            nodes = data.get("nodes", [])
            matrix = np.array(data["adjacency_matrix"])
            for i, src in enumerate(nodes):
                for j, tgt in enumerate(nodes):
                    if matrix[i, j] != 0:
                        graph.add_edge(src, tgt, float(matrix[i, j]))

        return graph


if __name__ == "__main__":
    print("Testing Causal Validator...")

    # Create default graph
    graph = CausalGraph.create_energy_graph()
    validator = CausalValidator(graph)

    # Test valid response
    print("\nTest 1: Valid response")
    response1 = (
        "The high outdoor temperature causes increased HVAC load, "
        "which leads to higher total consumption."
    )
    result1 = validator.validate(response1)
    print(f"  Is valid: {result1.is_valid}")
    print(f"  Score: {result1.score:.2f}")
    print(f"  Supported: {result1.supported_claims}")

    # Test invalid response (reversed causation)
    print("\nTest 2: Invalid response (reversed causation)")
    response2 = (
        "High electricity consumption causes people to leave the building, "
        "reducing occupancy."
    )
    result2 = validator.validate(response2)
    print(f"  Is valid: {result2.is_valid}")
    print(f"  Score: {result2.score:.2f}")
    print(f"  Violations: {result2.violations}")

    # Test direction mismatch
    print("\nTest 3: Direction mismatch")
    response3 = (
        "Higher occupancy decreases plug load."
    )
    result3 = validator.validate(response3)
    print(f"  Is valid: {result3.is_valid}")
    print(f"  Violations: {result3.violations}")

    # Save and load graph
    print("\nTest 4: Save/load graph")
    graph.save("/tmp/test_causal_graph.json")
    loaded = CausalGraph.load("/tmp/test_causal_graph.json")
    print(f"  Loaded {len(loaded.nodes)} nodes, {len(loaded.edges)} edges")

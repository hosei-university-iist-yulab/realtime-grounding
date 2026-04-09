"""
Unit tests for the causal validation module.

Run with: pytest tests/test_causal.py -v
"""

import pytest
import json
from pathlib import Path
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.causal import CausalGraph, CausalValidator, ValidationResult


@pytest.fixture
def simple_graph():
    """Create a simple test graph."""
    graph = CausalGraph()
    graph.add_edge("A", "B", strength=0.8)
    graph.add_edge("B", "C", strength=0.6)
    graph.add_edge("A", "C", strength=0.3)
    return graph


@pytest.fixture
def energy_graph():
    """Create energy domain graph."""
    return CausalGraph.create_energy_graph()


@pytest.fixture
def validator(energy_graph):
    """Create validator with energy graph."""
    return CausalValidator(energy_graph)


class TestCausalGraph:
    """Tests for CausalGraph."""

    def test_empty_graph(self):
        """Test empty graph creation."""
        graph = CausalGraph()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_add_edge(self, simple_graph):
        """Test adding edges."""
        assert simple_graph.has_edge("A", "B")
        assert simple_graph.has_edge("B", "C")
        assert not simple_graph.has_edge("C", "A")

    def test_get_causes(self, simple_graph):
        """Test getting causes of a node."""
        causes_c = simple_graph.get_causes("C")
        assert "B" in causes_c
        assert "A" in causes_c
        assert len(causes_c) == 2

    def test_get_effects(self, simple_graph):
        """Test getting effects of a node."""
        effects_a = simple_graph.get_effects("A")
        assert "B" in effects_a
        assert "C" in effects_a
        assert len(effects_a) == 2

    def test_get_ancestors(self, simple_graph):
        """Test getting all ancestors."""
        ancestors_c = simple_graph.get_ancestors("C")
        assert "A" in ancestors_c
        assert "B" in ancestors_c

    def test_get_descendants(self, simple_graph):
        """Test getting all descendants."""
        descendants_a = simple_graph.get_descendants("A")
        assert "B" in descendants_a
        assert "C" in descendants_a

    def test_valid_path(self, simple_graph):
        """Test path validation."""
        assert simple_graph.is_valid_path(["A", "B", "C"])
        assert simple_graph.is_valid_path(["A", "C"])
        assert not simple_graph.is_valid_path(["C", "B", "A"])

    def test_save_load(self, simple_graph):
        """Test save and load functionality."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            simple_graph.save(path)

            loaded = CausalGraph.load(path)

            assert loaded.has_edge("A", "B")
            assert loaded.has_edge("B", "C")
            assert len(loaded.nodes) == len(simple_graph.nodes)
            assert len(loaded.edges) == len(simple_graph.edges)
        finally:
            Path(path).unlink()

    def test_energy_graph_structure(self, energy_graph):
        """Test energy graph has expected structure."""
        # Should have key energy relationships
        assert energy_graph.has_edge("outdoor_temperature", "hvac_load")
        assert energy_graph.has_edge("occupancy", "plug_load")
        assert energy_graph.has_edge("hvac_load", "total_consumption")


class TestCausalValidator:
    """Tests for CausalValidator."""

    def test_valid_claim(self, validator):
        """Test validation of valid causal claim."""
        response = "High outdoor temperature causes increased HVAC load."
        result = validator.validate(response)

        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert len(result.violations) == 0

    def test_reversed_claim(self, validator):
        """Test detection of reversed causation."""
        response = "High electricity consumption causes the temperature to rise."
        result = validator.validate(response)

        # Should detect invalid relationship
        assert len(result.violations) > 0 or len(result.unsupported_claims) > 0

    def test_no_causal_claims(self, validator):
        """Test response with no causal claims."""
        response = "The building is currently using 150 kWh of electricity."
        result = validator.validate(response)

        # No claims to violate
        assert result.is_valid
        assert result.score == 1.0

    def test_multiple_claims(self, validator):
        """Test response with multiple claims."""
        response = (
            "Higher occupancy leads to increased plug load. "
            "Temperature affects HVAC consumption."
        )
        result = validator.validate(response)

        assert result.is_valid
        assert len(result.supported_claims) >= 1

    def test_suggestion_correction(self, validator):
        """Test correction suggestions."""
        suggestion = validator.suggest_correction("reversed causation detected")
        assert len(suggestion) > 0

    def test_enrich_with_causes(self, validator):
        """Test cause enrichment."""
        causes = validator.enrich_with_causes("hvac_load")
        assert len(causes) > 0
        assert any("temperature" in c for c in causes)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self):
        """Test valid result creation."""
        result = ValidationResult(
            is_valid=True,
            violations=[],
            score=1.0,
            supported_claims=["A causes B"],
            unsupported_claims=[]
        )
        assert result.is_valid
        assert result.score == 1.0

    def test_invalid_result(self):
        """Test invalid result creation."""
        result = ValidationResult(
            is_valid=False,
            violations=["Reversed: B does not cause A"],
            score=0.5,
            supported_claims=["C causes D"],
            unsupported_claims=["B causes A"]
        )
        assert not result.is_valid
        assert len(result.violations) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

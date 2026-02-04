"""
End-to-end test: JSON-RPC function call (from AGENTS.md spec).

This test validates generation with const constraints.
"""

import json
import pytest
from pathlib import Path


# Load schema fixture
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
with open(FIXTURES_DIR / "schemas" / "jsonrpc.json") as f:
    JSONRPC_SCHEMA = json.load(f)

with open(FIXTURES_DIR / "prompts" / "prompts.json") as f:
    PROMPTS = json.load(f)["jsonrpc"]


@pytest.mark.e2e
@pytest.mark.slow
class TestJSONRPC:
    """Test JSON-RPC generation (spec requirement)."""

    @pytest.fixture(scope="class")
    def generator(self):
        """Create generator fixture."""
        from grammar_guard import GrammarConstrainedGenerator

        # Force CPU to avoid device-specific issues during testing
        return GrammarConstrainedGenerator(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            backend="transformers",
            device="cpu"
        )

    def test_jsonrpc_call(self, generator):
        """Test generating JSON-RPC call."""
        result = generator.generate(
            prompt=PROMPTS[0],  # "Generate a JSON-RPC call to method 'getUserProfile'..."
            schema=JSONRPC_SCHEMA,
            max_tokens=150,
            max_retries=3
        )

        assert result.output

        # Parse JSON
        try:
            parsed = json.loads(result.output)
        except json.JSONDecodeError as e:
            pytest.fail(f"Generated invalid JSON: {e}\nOutput: {result.output}")

        # Required fields
        assert "jsonrpc" in parsed, "Missing required field: jsonrpc"
        assert "method" in parsed, "Missing required field: method"
        assert "id" in parsed, "Missing required field: id"

        # jsonrpc version must be "2.0" (const constraint)
        assert parsed["jsonrpc"] == "2.0", f"jsonrpc should be '2.0', got {parsed['jsonrpc']}"

        # Type validation
        assert isinstance(parsed["method"], str), "method should be string"
        assert len(parsed["method"]) >= 1, "method cannot be empty"
        assert isinstance(parsed["id"], int), "id should be integer"

        # Optional params
        if "params" in parsed:
            assert isinstance(parsed["params"], dict), "params should be object"

        print(f"✓ Generated valid JSON-RPC call: {parsed['method']}")

    def test_jsonrpc_with_params(self, generator):
        """Test generating JSON-RPC call with params."""
        result = generator.generate(
            prompt=PROMPTS[2],  # "Make a JSON-RPC call to 'getData' with id 123"
            schema=JSONRPC_SCHEMA,
            max_tokens=150,
            max_retries=3
        )

        assert result.output

        parsed = json.loads(result.output)

        # Validate structure
        assert parsed["jsonrpc"] == "2.0"
        assert "method" in parsed
        assert "id" in parsed

        print(f"✓ Generated JSON-RPC with id {parsed['id']}")

    def test_multiple_jsonrpc_calls(self, generator):
        """Test generating multiple JSON-RPC calls."""
        valid_count = 0

        for prompt in PROMPTS:
            result = generator.generate(
                prompt=prompt,
                schema=JSONRPC_SCHEMA,
                max_tokens=150,
                max_retries=2
            )

            if result.is_valid:
                parsed = json.loads(result.output)
                assert parsed["jsonrpc"] == "2.0"
                valid_count += 1

        # At least 2/3 should succeed
        assert valid_count >= 2, f"Only {valid_count}/3 generations succeeded"

        print(f"✓ Generated {valid_count}/3 valid JSON-RPC calls")

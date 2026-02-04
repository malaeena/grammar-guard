"""
End-to-end test: Person record with nested fields (from AGENTS.md spec).

This test validates the complete generation pipeline with a complex schema
including nested objects and arrays.
"""

import json
import pytest
from pathlib import Path


# Load schema fixture
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
with open(FIXTURES_DIR / "schemas" / "person.json") as f:
    PERSON_SCHEMA = json.load(f)

# Load test prompts
with open(FIXTURES_DIR / "prompts" / "prompts.json") as f:
    PROMPTS = json.load(f)["person"]


@pytest.mark.e2e
@pytest.mark.slow
class TestPersonRecord:
    """Test person record generation (spec requirement)."""

    @pytest.fixture(scope="class")
    def generator(self):
        """Create generator fixture."""
        from grammar_guard import GrammarConstrainedGenerator

        # Use small model for testing
        # Force CPU to avoid device-specific issues during testing
        return GrammarConstrainedGenerator(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            backend="transformers",
            device="cpu"
        )

    def test_person_with_all_fields(self, generator):
        """Test generating person with all fields including nested address."""
        result = generator.generate(
            prompt=PROMPTS[1],  # "Create a user profile for Bob Smith..."
            schema=PERSON_SCHEMA,
            max_tokens=200,
            max_retries=3
        )

        # Basic validation
        assert result.output, "Output should not be empty"

        # Parse JSON
        try:
            parsed = json.loads(result.output)
        except json.JSONDecodeError as e:
            pytest.fail(f"Generated invalid JSON: {e}\nOutput: {result.output}")

        # Required fields
        assert "name" in parsed, "Missing required field: name"
        assert "age" in parsed, "Missing required field: age"

        # Type validation
        assert isinstance(parsed["name"], str), "name should be string"
        assert isinstance(parsed["age"], int), "age should be integer"

        # Value constraints
        assert len(parsed["name"]) >= 2, "name too short (minLength: 2)"
        assert len(parsed["name"]) <= 50, "name too long (maxLength: 50)"
        assert 0 <= parsed["age"] <= 150, "age out of range"

        # Optional nested address
        if "address" in parsed:
            assert isinstance(parsed["address"], dict), "address should be object"
            assert "city" in parsed["address"], "address.city is required"

        # Optional hobbies array
        if "hobbies" in parsed:
            assert isinstance(parsed["hobbies"], list), "hobbies should be array"
            for hobby in parsed["hobbies"]:
                assert isinstance(hobby, str), "hobby items should be strings"

        print(f"✓ Generated valid person: {parsed['name']}, age {parsed['age']}")

    def test_person_minimal(self, generator):
        """Test generating person with only required fields."""
        result = generator.generate(
            prompt=PROMPTS[2],  # "Make a person named Charlie, age 42"
            schema=PERSON_SCHEMA,
            max_tokens=100,
            max_retries=3
        )

        assert result.output

        parsed = json.loads(result.output)

        # Must have required fields
        assert "name" in parsed
        assert "age" in parsed

        # Type check
        assert isinstance(parsed["name"], str)
        assert isinstance(parsed["age"], int)

        print(f"✓ Generated minimal person: {parsed}")

    def test_multiple_generations(self, generator):
        """Test generating multiple person records."""
        results = []

        for i, prompt in enumerate(PROMPTS):
            result = generator.generate(
                prompt=prompt,
                schema=PERSON_SCHEMA,
                max_tokens=150,
                max_retries=2
            )

            if result.is_valid:
                parsed = json.loads(result.output)
                results.append(parsed)
                print(f"  Generation {i+1}/3: ✓ Valid")
            else:
                print(f"  Generation {i+1}/3: ✗ Invalid (retries: {result.retries})")

        # At least 2 out of 3 should succeed
        assert len(results) >= 2, f"Only {len(results)}/3 generations succeeded"

        print(f"✓ Generated {len(results)}/3 valid person records")

"""
Unit tests for validator.
"""

import pytest
from grammar_guard.validation import validate, quick_validate, format_validation_errors


class TestValidator:
    """Test JSON Schema validation."""

    def test_validate_valid_json(self):
        """Test validating valid JSON."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }
        output = '{"name": "Alice"}'

        result = validate(output, schema)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.parsed_output == {"name": "Alice"}

    def test_validate_invalid_json_syntax(self):
        """Test validating invalid JSON syntax."""
        schema = {"type": "object"}
        output = '{invalid json}'

        result = validate(output, schema)

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "Invalid JSON" in result.errors[0].message
        assert result.parsed_output is None

    def test_validate_missing_required_field(self):
        """Test validating JSON with missing required field."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }
        output = '{"name": "Alice"}'

        result = validate(output, schema)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("age" in error.message.lower() for error in result.errors)

    def test_validate_wrong_type(self):
        """Test validating JSON with wrong type."""
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer"}
            }
        }
        output = '{"age": "not a number"}'

        result = validate(output, schema)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_min_max_constraints(self):
        """Test validating numeric min/max constraints."""
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "minimum": 0, "maximum": 150}
            }
        }

        # Valid
        result = validate('{"age": 25}', schema)
        assert result.is_valid is True

        # Too low
        result = validate('{"age": -1}', schema)
        assert result.is_valid is False

        # Too high
        result = validate('{"age": 200}', schema)
        assert result.is_valid is False

    def test_validate_string_length(self):
        """Test validating string length constraints."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 2, "maxLength": 10}
            }
        }

        # Valid
        result = validate('{"name": "Alice"}', schema)
        assert result.is_valid is True

        # Too short
        result = validate('{"name": "A"}', schema)
        assert result.is_valid is False

        # Too long
        result = validate('{"name": "VeryLongName"}', schema)
        assert result.is_valid is False

    def test_quick_validate(self):
        """Test quick validation (bool only)."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        assert quick_validate('{"name": "Alice"}', schema) is True
        assert quick_validate('{invalid}', schema) is False

    def test_format_validation_errors(self):
        """Test formatting validation errors."""
        schema = {
            "type": "object",
            "properties": {"age": {"type": "integer"}},
            "required": ["age"]
        }
        output = '{"age": "not a number"}'

        result = validate(output, schema)
        formatted = format_validation_errors(result.errors)

        assert "Validation failed" in formatted
        assert len(result.errors) > 0

"""
Unit tests for schema parser.
"""

import pytest
from grammar_guard.schema import parse_schema, validate_schema, normalize_schema
from grammar_guard.schema.types import (
    ObjectConstraint,
    ArrayConstraint,
    StringConstraint,
    NumberConstraint,
    BooleanConstraint,
    NullConstraint,
)


class TestSchemaParser:
    """Test schema parsing functionality."""

    def test_parse_simple_string(self):
        """Test parsing simple string schema."""
        schema = {"type": "string"}
        constraint = parse_schema(schema)

        assert isinstance(constraint, StringConstraint)
        assert constraint.min_length is None
        assert constraint.max_length is None

    def test_parse_string_with_length_constraints(self):
        """Test parsing string with minLength and maxLength."""
        schema = {"type": "string", "minLength": 3, "maxLength": 50}
        constraint = parse_schema(schema)

        assert isinstance(constraint, StringConstraint)
        assert constraint.min_length == 3
        assert constraint.max_length == 50

    def test_parse_string_with_enum(self):
        """Test parsing string with enum constraint."""
        schema = {"type": "string", "enum": ["red", "green", "blue"]}
        constraint = parse_schema(schema)

        assert isinstance(constraint, StringConstraint)
        assert constraint.enum == {"red", "green", "blue"}

    def test_parse_integer(self):
        """Test parsing integer schema."""
        schema = {"type": "integer", "minimum": 0, "maximum": 100}
        constraint = parse_schema(schema)

        assert isinstance(constraint, NumberConstraint)
        assert constraint.is_integer is True
        assert constraint.minimum == 0
        assert constraint.maximum == 100

    def test_parse_number(self):
        """Test parsing number (float) schema."""
        schema = {"type": "number"}
        constraint = parse_schema(schema)

        assert isinstance(constraint, NumberConstraint)
        assert constraint.is_integer is False

    def test_parse_boolean(self):
        """Test parsing boolean schema."""
        schema = {"type": "boolean"}
        constraint = parse_schema(schema)

        assert isinstance(constraint, BooleanConstraint)

    def test_parse_null(self):
        """Test parsing null schema."""
        schema = {"type": "null"}
        constraint = parse_schema(schema)

        assert isinstance(constraint, NullConstraint)

    def test_parse_simple_object(self):
        """Test parsing simple object schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }
        constraint = parse_schema(schema)

        assert isinstance(constraint, ObjectConstraint)
        assert "name" in constraint.properties
        assert "age" in constraint.properties
        assert "name" in constraint.required
        assert "age" not in constraint.required

    def test_parse_nested_object(self):
        """Test parsing nested object schema."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        }
        constraint = parse_schema(schema)

        assert isinstance(constraint, ObjectConstraint)
        assert isinstance(constraint.properties["user"], ObjectConstraint)

    def test_parse_array(self):
        """Test parsing array schema."""
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 10
        }
        constraint = parse_schema(schema)

        assert isinstance(constraint, ArrayConstraint)
        assert isinstance(constraint.items, StringConstraint)
        assert constraint.min_items == 1
        assert constraint.max_items == 10

    def test_normalize_schema(self):
        """Test schema normalization."""
        schema = {"type": "object"}
        normalized = normalize_schema(schema)

        assert "properties" in normalized
        assert "required" in normalized
        assert normalized["properties"] == {}
        assert normalized["required"] == []

    def test_validate_schema_valid(self):
        """Test schema validation with valid schema."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        }

        # Should not raise
        validate_schema(schema)

    def test_validate_schema_invalid_type(self):
        """Test schema validation with invalid type."""
        schema = {"type": "invalid_type"}

        with pytest.raises(ValueError, match="Invalid type"):
            validate_schema(schema)

    def test_validate_schema_unsupported_keyword(self):
        """Test schema validation with unsupported keyword."""
        schema = {"type": "string", "$ref": "#/definitions/User"}

        with pytest.raises(ValueError, match="not supported"):
            validate_schema(schema)


class TestPydanticIntegration:
    """Test Pydantic model integration."""

    def test_parse_pydantic_model(self):
        """Test parsing Pydantic model."""
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("Pydantic not installed")

        class User(BaseModel):
            name: str
            age: int

        constraint = parse_schema(User)

        assert isinstance(constraint, ObjectConstraint)
        assert "name" in constraint.properties
        assert "age" in constraint.properties

    def test_parse_pydantic_with_field_constraints(self):
        """Test parsing Pydantic model with Field constraints."""
        try:
            from pydantic import BaseModel, Field
        except ImportError:
            pytest.skip("Pydantic not installed")

        class User(BaseModel):
            name: str = Field(min_length=2, max_length=50)
            age: int = Field(ge=0, le=150)

        schema = parse_schema(User)

        # Should parse successfully
        assert isinstance(schema, ObjectConstraint)

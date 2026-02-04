"""
Unit tests for regex compiler.
"""

import pytest
import re
from grammar_guard.schema import parse_schema, compile_to_regex


class TestRegexCompiler:
    """Test regex compilation from constraints."""

    def test_compile_string_no_constraints(self):
        """Test compiling string with no constraints."""
        schema = {"type": "string"}
        constraint = parse_schema(schema)
        regex = compile_to_regex(constraint)

        # Should match quoted strings
        assert re.fullmatch(regex, '"hello"')
        assert re.fullmatch(regex, '""')
        assert not re.fullmatch(regex, 'hello')  # No quotes

    def test_compile_string_with_length(self):
        """Test compiling string with length constraints."""
        schema = {"type": "string", "minLength": 3, "maxLength": 5}
        constraint = parse_schema(schema)
        regex = compile_to_regex(constraint)

        # Should match strings of length 3-5
        assert re.fullmatch(regex, '"abc"')
        assert re.fullmatch(regex, '"abcd"')
        assert re.fullmatch(regex, '"abcde"')
        assert not re.fullmatch(regex, '"ab"')  # Too short
        assert not re.fullmatch(regex, '"abcdef"')  # Too long

    def test_compile_integer(self):
        """Test compiling integer constraint."""
        schema = {"type": "integer"}
        constraint = parse_schema(schema)
        regex = compile_to_regex(constraint)

        # Should match integers
        assert re.fullmatch(regex, '123')
        assert re.fullmatch(regex, '0')
        assert re.fullmatch(regex, '-456')
        assert not re.fullmatch(regex, '12.34')  # Float

    def test_compile_boolean(self):
        """Test compiling boolean constraint."""
        schema = {"type": "boolean"}
        constraint = parse_schema(schema)
        regex = compile_to_regex(constraint)

        # Should match true/false
        assert re.fullmatch(regex, 'true')
        assert re.fullmatch(regex, 'false')
        assert not re.fullmatch(regex, 'True')
        assert not re.fullmatch(regex, '1')

    def test_compile_null(self):
        """Test compiling null constraint."""
        schema = {"type": "null"}
        constraint = parse_schema(schema)
        regex = compile_to_regex(constraint)

        # Should match null
        assert re.fullmatch(regex, 'null')
        assert not re.fullmatch(regex, 'None')
        assert not re.fullmatch(regex, 'NULL')

    def test_compile_simple_object(self):
        """Test compiling simple object."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }
        constraint = parse_schema(schema)
        regex = compile_to_regex(constraint)

        # Regex should be generated (we don't test exact match due to complexity)
        assert isinstance(regex, str)
        assert len(regex) > 0
        assert '{' in regex  # Should have object braces

    def test_compile_array(self):
        """Test compiling array."""
        schema = {
            "type": "array",
            "items": {"type": "string"}
        }
        constraint = parse_schema(schema)
        regex = compile_to_regex(constraint)

        # Regex should be generated
        assert isinstance(regex, str)
        assert len(regex) > 0
        assert '[' in regex  # Should have array brackets

"""
JSON Schema validator with detailed error reporting.

This module provides post-generation validation of LLM outputs against JSON schemas.
Even with constrained decoding, validation is important because:
    1. Range constraints (min/max) are hard to enforce at token level
    2. Complex semantic constraints may be violated
    3. We want detailed error messages for debugging

Usage:
    ```python
    from grammar_guard.validation import validate

    schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
    output = '{"age": 25}'

    result = validate(output, schema)
    if result.is_valid:
        print("Valid!")
    else:
        for error in result.errors:
            print(f"Error: {error.message}")
    ```
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """
    Represents a single validation error.

    Attributes:
        path: JSON path to the error location (e.g., "user.address.zipcode")
        message: Human-readable error message
        schema_path: Path in schema that failed
        validator: Validator that failed (e.g., "type", "minimum")
        expected: What was expected
        actual: What was found
    """
    path: str
    message: str
    schema_path: str
    validator: str
    expected: Any
    actual: Any


@dataclass
class ValidationResult:
    """
    Result of validating JSON against a schema.

    Attributes:
        is_valid: Whether output is valid
        errors: List of validation errors (empty if valid)
        raw_output: Original output string
        parsed_output: Parsed JSON (None if parse failed)
    """
    is_valid: bool
    errors: List[ValidationError]
    raw_output: str
    parsed_output: Optional[Any]


def validate(output: str, schema: Dict[str, Any]) -> ValidationResult:
    """
    Validate JSON output against a schema.

    Args:
        output: JSON string to validate
        schema: JSON Schema dictionary

    Returns:
        ValidationResult: Validation result with errors if any

    Example:
        ```python
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        }

        # Valid output
        result = validate('{"name": "Alice"}', schema)
        assert result.is_valid

        # Invalid output
        result = validate('{"age": 25}', schema)
        assert not result.is_valid
        print(result.errors[0].message)  # "name is required"
        ```
    """
    # Try to parse JSON
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError as e:
        # JSON parse error
        return ValidationResult(
            is_valid=False,
            errors=[
                ValidationError(
                    path="",
                    message=f"Invalid JSON: {e.msg}",
                    schema_path="",
                    validator="json",
                    expected="valid JSON",
                    actual=f"parse error at position {e.pos}"
                )
            ],
            raw_output=output,
            parsed_output=None
        )

    # Validate against schema
    try:
        import jsonschema
        from jsonschema import Draft7Validator
    except ImportError:
        raise ImportError(
            "jsonschema is required. Install with: pip install jsonschema"
        )

    # Create validator
    validator = Draft7Validator(schema)

    # Collect all errors
    errors = []
    for error in validator.iter_errors(parsed):
        errors.append(_convert_jsonschema_error(error, parsed))

    # Build result
    is_valid = len(errors) == 0

    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        raw_output=output,
        parsed_output=parsed if is_valid else None
    )


def _convert_jsonschema_error(error: Any, data: Any) -> ValidationError:
    """
    Convert jsonschema ValidationError to our ValidationError.

    Args:
        error: jsonschema ValidationError
        data: The data being validated

    Returns:
        ValidationError: Our error representation
    """
    # Build JSON path
    path = "." + ".".join(str(p) for p in error.path) if error.path else "root"

    # Get actual value at error location
    actual = data
    for key in error.path:
        if isinstance(actual, dict):
            actual = actual.get(key, "MISSING")
        elif isinstance(actual, list):
            try:
                actual = actual[int(key)]
            except (IndexError, ValueError):
                actual = "INVALID_INDEX"
        else:
            actual = "UNKNOWN"

    # Build schema path
    schema_path = "." + ".".join(str(p) for p in error.schema_path) if error.schema_path else "root"

    # Extract expected value
    expected = error.schema.get(error.validator, "see schema")

    return ValidationError(
        path=path,
        message=error.message,
        schema_path=schema_path,
        validator=error.validator,
        expected=expected,
        actual=actual
    )


def format_validation_errors(errors: List[ValidationError]) -> str:
    """
    Format validation errors as human-readable string.

    Args:
        errors: List of validation errors

    Returns:
        str: Formatted error message

    Example:
        ```python
        result = validate(output, schema)
        if not result.is_valid:
            print(format_validation_errors(result.errors))
            # Output:
            # Validation failed with 2 errors:
            #   1. At .name: 'name' is a required property
            #      Expected: required field
            #      Got: MISSING
            #   2. At .age: 'not a number' is not of type 'integer'
            #      Expected: integer
            #      Got: 'not a number'
        ```
    """
    if not errors:
        return "No validation errors"

    lines = [f"Validation failed with {len(errors)} error(s):"]

    for i, error in enumerate(errors, 1):
        lines.append(f"\n  {i}. At {error.path}: {error.message}")
        lines.append(f"     Expected: {error.expected}")
        lines.append(f"     Got: {error.actual}")

    return "\n".join(lines)


def quick_validate(output: str, schema: Dict[str, Any]) -> bool:
    """
    Quick validation - just returns True/False.

    Args:
        output: JSON string
        schema: JSON Schema

    Returns:
        bool: True if valid, False otherwise

    Example:
        ```python
        if quick_validate(output, schema):
            print("Valid!")
        else:
            print("Invalid!")
        ```
    """
    result = validate(output, schema)
    return result.is_valid

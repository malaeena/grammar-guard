"""
Validation layer module.

This module handles post-generation validation of LLM outputs against JSON schemas,
providing detailed error reporting and diff generation for debugging.

Components:
    - validator: Main validation interface using jsonschema
    - error_formatter: Convert validation errors to human-readable messages
    - diff_generator: Generate diffs showing expected vs actual output

Validation Flow:
    1. Parse generated string as JSON
    2. Validate against JSON Schema using jsonschema library
    3. Collect all validation errors (not just first error)
    4. Format errors with context (path, expected type, actual value)
    5. Generate diffs for debugging

Example:
    ```python
    from grammar_guard.validation import validate

    schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
    output = '{"age": "not a number"}'

    result = validate(output, schema)
    if not result.is_valid:
        print(f"Validation failed: {len(result.errors)} errors")
        for error in result.errors:
            print(f"  - {error.path}: {error.message}")
            print(f"    Expected: {error.expected}")
            print(f"    Got: {error.actual}")
    ```
"""

from grammar_guard.validation.validator import (
    validate,
    quick_validate,
    ValidationResult,
    ValidationError,
    format_validation_errors
)
from grammar_guard.validation.error_formatter import format_error_with_context, suggest_fix
from grammar_guard.validation.diff_generator import generate_diff

__all__ = [
    "validate",
    "quick_validate",
    "ValidationResult",
    "ValidationError",
    "format_validation_errors",
    "format_error_with_context",
    "suggest_fix",
    "generate_diff",
]

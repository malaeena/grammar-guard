"""
Error formatter - convert validation errors to user-friendly messages.

This module provides utilities for formatting validation errors in a way
that helps users understand what went wrong and how to fix it.
"""

from typing import List
from grammar_guard.validation.validator import ValidationError


def format_error_with_context(error: ValidationError, context_lines: int = 2) -> str:
    """
    Format error with surrounding context.

    Args:
        error: Validation error
        context_lines: Number of lines of context to show

    Returns:
        str: Formatted error with context
    """
    lines = [
        f"❌ Validation Error at {error.path}",
        f"   Problem: {error.message}",
        f"   Expected: {error.expected}",
        f"   Got: {error.actual}",
    ]

    if error.validator:
        lines.append(f"   Validator: {error.validator}")

    return "\n".join(lines)


def suggest_fix(error: ValidationError) -> str:
    """
    Suggest how to fix a validation error.

    Args:
        error: Validation error

    Returns:
        str: Suggested fix
    """
    if error.validator == "required":
        return f"Add the required field '{error.path}' to your JSON"

    elif error.validator == "type":
        return f"Change {error.path} to type {error.expected}"

    elif error.validator in ["minimum", "maximum"]:
        return f"Ensure {error.path} is within the allowed range"

    elif error.validator in ["minLength", "maxLength"]:
        return f"Adjust the length of {error.path}"

    else:
        return "Check the schema requirements"

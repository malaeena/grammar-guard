"""
Diff generator - show differences between expected and actual output.

This module provides utilities for generating diffs that highlight
what's wrong with the generated output.
"""

import json
from typing import Any, Dict


def generate_diff(expected_schema: Dict, actual_data: Any) -> str:
    """
    Generate a diff showing what's missing or wrong.

    Args:
        expected_schema: JSON Schema
        actual_data: Actual generated data

    Returns:
        str: Diff description
    """
    lines = ["Expected vs Actual:"]

    # For MVP, just show JSON comparison
    if expected_schema.get("type") == "object":
        expected_props = expected_schema.get("properties", {})
        required = set(expected_schema.get("required", []))

        if isinstance(actual_data, dict):
            # Check required fields
            for field in required:
                if field in actual_data:
                    lines.append(f"  ✓ {field}: present")
                else:
                    lines.append(f"  ✗ {field}: MISSING (required)")

            # Check extra fields
            for field in actual_data:
                if field not in expected_props:
                    lines.append(f"  ? {field}: unexpected field")
        else:
            lines.append(f"  ✗ Expected object, got {type(actual_data).__name__}")

    return "\n".join(lines)

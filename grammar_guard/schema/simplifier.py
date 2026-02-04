"""
Schema simplifier - progressively relax schema constraints for retry logic.

When LLM generation fails to produce valid output even with constrained decoding,
we progressively simplify the schema to give the model more freedom. This module
implements the simplification strategy.

Simplification Levels:
    Level 0: Original schema (no simplification)
    Level 1: Remove optional fields (keep only required fields)
    Level 2: Relax length constraints by 50% (strings, arrays)
    Level 3: Relax numeric constraints by 2x
    Level 4: Remove enum constraints
    Level 5: Maximum simplification (remove most constraints)

Strategy:
    Start with level 0, increment on each retry. The constraint types themselves
    know how to simplify (via the simplify() method), so this module primarily
    coordinates the simplification process and tracks what was changed.

Usage:
    ```python
    from grammar_guard.schema import parse_schema, simplify_schema

    original_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 3, "maxLength": 50},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "email": {"type": "string"}  # Optional
        },
        "required": ["name", "age"]
    }

    # Level 1: Remove optional fields
    simplified = simplify_schema(original_schema, level=1)
    # Result: Only "name" and "age" remain

    # Level 2: Also relax length constraints
    simplified = simplify_schema(original_schema, level=2)
    # Result: minLength=1, maxLength=75 for name

    # Level 5: Maximum simplification
    simplified = simplify_schema(original_schema, level=5)
    # Result: name and age with minimal constraints
    ```
"""

from typing import Any, Dict, List, Tuple

from grammar_guard.schema.parser import parse_schema
from grammar_guard.schema.types import ConstraintNode


def simplify_schema(
    schema: Dict[str, Any],
    level: int,
    track_changes: bool = True
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Simplify a JSON Schema to a given level.

    Args:
        schema: Original JSON Schema dictionary
        level: Simplification level (0-5, higher = more simplified)
            0: No simplification
            1: Remove optional fields
            2: Relax length constraints (50% wider)
            3: Relax numeric constraints (2x wider)
            4: Remove enum constraints
            5: Maximum simplification
        track_changes: If True, return list of changes made

    Returns:
        Tuple of (simplified_schema, changes_list)
        - simplified_schema: Simplified JSON Schema dictionary
        - changes_list: List of human-readable descriptions of changes

    Example:
        ```python
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 5},
                "age": {"type": "integer"},
                "email": {"type": "string"}
            },
            "required": ["name"]
        }

        simplified, changes = simplify_schema(schema, level=2)
        # simplified: name field has minLength=2 or 3
        # changes: ["Removed optional field: email", "Relaxed minLength on name: 5 → 2"]
        ```
    """
    if level == 0:
        return schema, []

    # Parse schema to constraint tree
    constraint = parse_schema(schema)

    # Simplify constraint
    simplified_constraint = constraint.simplify(level)

    # Convert back to JSON Schema dict
    simplified_schema = _constraint_to_schema(simplified_constraint)

    # Track changes if requested
    changes = []
    if track_changes:
        changes = _describe_changes(schema, simplified_schema, level)

    return simplified_schema, changes


def _constraint_to_schema(constraint: ConstraintNode) -> Dict[str, Any]:
    """
    Convert a ConstraintNode back to a JSON Schema dictionary.

    This is the inverse of parse_schema() - it reconstructs a JSON Schema
    from our internal constraint representation.

    Args:
        constraint: Constraint tree to convert

    Returns:
        Dict: JSON Schema dictionary

    Note:
        This conversion may not be perfect - some information may be lost
        or represented differently than the original schema.
    """
    from grammar_guard.schema.types import (
        ArrayConstraint,
        BooleanConstraint,
        NullConstraint,
        NumberConstraint,
        ObjectConstraint,
        StringConstraint,
        UnionConstraint,
    )

    if isinstance(constraint, ObjectConstraint):
        schema = {
            "type": "object",
            "properties": {
                k: _constraint_to_schema(v) for k, v in constraint.properties.items()
            },
        }
        if constraint.required:
            schema["required"] = list(constraint.required)
        if constraint.additional_properties:
            schema["additionalProperties"] = True
        return schema

    elif isinstance(constraint, ArrayConstraint):
        schema = {"type": "array"}
        if constraint.items:
            schema["items"] = _constraint_to_schema(constraint.items)
        if constraint.min_items is not None:
            schema["minItems"] = constraint.min_items
        if constraint.max_items is not None:
            schema["maxItems"] = constraint.max_items
        return schema

    elif isinstance(constraint, StringConstraint):
        schema = {"type": "string"}
        if constraint.min_length is not None:
            schema["minLength"] = constraint.min_length
        if constraint.max_length is not None:
            schema["maxLength"] = constraint.max_length
        if constraint.pattern is not None:
            schema["pattern"] = constraint.pattern
        if constraint.enum is not None:
            schema["enum"] = list(constraint.enum)
        return schema

    elif isinstance(constraint, NumberConstraint):
        schema = {"type": "integer" if constraint.is_integer else "number"}
        if constraint.minimum is not None:
            schema["minimum"] = constraint.minimum
        if constraint.maximum is not None:
            schema["maximum"] = constraint.maximum
        if constraint.enum is not None:
            schema["enum"] = list(constraint.enum)
        return schema

    elif isinstance(constraint, BooleanConstraint):
        return {"type": "boolean"}

    elif isinstance(constraint, NullConstraint):
        return {"type": "null"}

    elif isinstance(constraint, UnionConstraint):
        return {
            "anyOf": [_constraint_to_schema(opt) for opt in constraint.options]
        }

    else:
        # Unknown constraint type
        return {}


def _describe_changes(
    original: Dict[str, Any],
    simplified: Dict[str, Any],
    level: int
) -> List[str]:
    """
    Generate human-readable descriptions of changes made during simplification.

    Args:
        original: Original JSON Schema
        simplified: Simplified JSON Schema
        level: Simplification level used

    Returns:
        List of change descriptions

    Example:
        ["Removed optional field: email",
         "Relaxed minLength on name: 5 → 2",
         "Relaxed maximum on age: 150 → 300"]
    """
    changes = []

    # Level 1: Removed optional fields
    if level >= 1:
        orig_props = set(original.get("properties", {}).keys())
        simp_props = set(simplified.get("properties", {}).keys())
        removed = orig_props - simp_props
        for field in removed:
            changes.append(f"Removed optional field: {field}")

    # Level 2+: Relaxed length constraints
    if level >= 2:
        for field in simplified.get("properties", {}):
            if field not in original.get("properties", {}):
                continue

            orig_field = original["properties"][field]
            simp_field = simplified["properties"][field]

            # String length constraints
            if orig_field.get("type") == "string":
                if "minLength" in orig_field and "minLength" in simp_field:
                    if orig_field["minLength"] != simp_field["minLength"]:
                        changes.append(
                            f"Relaxed minLength on {field}: "
                            f"{orig_field['minLength']} → {simp_field['minLength']}"
                        )
                if "maxLength" in orig_field and "maxLength" in simp_field:
                    if orig_field["maxLength"] != simp_field["maxLength"]:
                        changes.append(
                            f"Relaxed maxLength on {field}: "
                            f"{orig_field['maxLength']} → {simp_field['maxLength']}"
                        )

            # Array constraints
            if orig_field.get("type") == "array":
                if "minItems" in orig_field and "minItems" in simp_field:
                    if orig_field["minItems"] != simp_field["minItems"]:
                        changes.append(
                            f"Relaxed minItems on {field}: "
                            f"{orig_field['minItems']} → {simp_field['minItems']}"
                        )
                if "maxItems" in orig_field and "maxItems" in simp_field:
                    if orig_field["maxItems"] != simp_field["maxItems"]:
                        changes.append(
                            f"Relaxed maxItems on {field}: "
                            f"{orig_field['maxItems']} → {simp_field['maxItems']}"
                        )

    # Level 3+: Relaxed numeric constraints
    if level >= 3:
        for field in simplified.get("properties", {}):
            if field not in original.get("properties", {}):
                continue

            orig_field = original["properties"][field]
            simp_field = simplified["properties"][field]

            if orig_field.get("type") in ["integer", "number"]:
                if "minimum" in orig_field and "minimum" in simp_field:
                    if orig_field["minimum"] != simp_field["minimum"]:
                        changes.append(
                            f"Relaxed minimum on {field}: "
                            f"{orig_field['minimum']} → {simp_field['minimum']}"
                        )
                if "maximum" in orig_field and "maximum" in simp_field:
                    if orig_field["maximum"] != simp_field["maximum"]:
                        changes.append(
                            f"Relaxed maximum on {field}: "
                            f"{orig_field['maximum']} → {simp_field['maximum']}"
                        )

    # Level 4+: Removed enum constraints
    if level >= 4:
        for field in simplified.get("properties", {}):
            if field not in original.get("properties", {}):
                continue

            orig_field = original["properties"][field]
            simp_field = simplified["properties"][field]

            if "enum" in orig_field and "enum" not in simp_field:
                changes.append(f"Removed enum constraint on {field}")

    # General message
    if level == 5:
        changes.append("Applied maximum simplification (level 5)")

    return changes


def get_simplification_summary(level: int) -> str:
    """
    Get a human-readable summary of what a simplification level does.

    Args:
        level: Simplification level (0-5)

    Returns:
        str: Description of simplification at this level

    Example:
        ```python
        print(get_simplification_summary(2))
        # "Level 2: Remove optional fields + relax length constraints by 50%"
        ```
    """
    summaries = {
        0: "Level 0: No simplification (original schema)",
        1: "Level 1: Remove optional fields (keep only required)",
        2: "Level 2: Remove optional fields + relax length constraints by 50%",
        3: "Level 3: Remove optional fields + relax all constraints by 2x",
        4: "Level 4: Remove optional fields + remove enum constraints",
        5: "Level 5: Maximum simplification (minimal constraints)"
    }
    return summaries.get(level, f"Level {level}: Unknown")


def suggest_next_level(current_level: int, max_level: int = 5) -> int:
    """
    Suggest the next simplification level to try.

    Args:
        current_level: Current simplification level
        max_level: Maximum allowed level (default: 5)

    Returns:
        int: Next level to try, or -1 if at maximum

    Example:
        ```python
        next_level = suggest_next_level(current_level=2)
        # next_level = 3
        ```
    """
    if current_level >= max_level:
        return -1
    return current_level + 1


def should_simplify(
    validation_errors: List[Any],
    current_level: int,
    max_retries: int
) -> bool:
    """
    Determine if we should simplify the schema based on validation errors.

    This uses heuristics to decide if simplification would help:
    - If there are many validation errors, simplification may help
    - If we're early in retry attempts, try simplification
    - If errors are about missing required fields, simplification won't help

    Args:
        validation_errors: List of validation errors from previous attempt
        current_level: Current simplification level
        max_retries: Maximum number of retries allowed

    Returns:
        bool: True if we should try simplifying the schema

    Example:
        ```python
        errors = [...]  # Validation errors
        if should_simplify(errors, current_level=1, max_retries=3):
            # Try next simplification level
            next_level = current_level + 1
        ```
    """
    # Always allow at least one simplification attempt
    if current_level == 0:
        return True

    # Don't exceed max retries
    if current_level >= max_retries:
        return False

    # If we have validation errors, simplification might help
    if len(validation_errors) > 0:
        return True

    return False

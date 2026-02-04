"""
JSON Schema parser - converts JSON Schema dicts to internal constraint representation.

This module is the main entry point for parsing JSON schemas into the internal
constraint tree structure. It handles:
    - JSON Schema Draft 7 features (subset for MVP)
    - Type normalization and validation
    - Nested object and array structures
    - Validation constraints (minLength, maximum, etc.)

Usage:
    ```python
    from grammar_guard.schema import parse_schema

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 3},
            "age": {"type": "integer", "minimum": 0}
        },
        "required": ["name"]
    }

    constraint = parse_schema(schema)
    regex = constraint.to_regex()
    ```
"""

from typing import Any, Dict, List, Optional, Set, Union

from grammar_guard.schema.types import (
    ArrayConstraint,
    BooleanConstraint,
    ConstraintNode,
    NullConstraint,
    NumberConstraint,
    ObjectConstraint,
    StringConstraint,
    UnionConstraint,
)


def parse_schema(schema: Union[Dict[str, Any], type]) -> ConstraintNode:
    """
    Parse a JSON Schema or Pydantic model into a ConstraintNode tree.

    This is the main entry point for schema parsing. It accepts either:
    - A JSON Schema dictionary
    - A Pydantic BaseModel class (converted to JSON Schema internally)

    Args:
        schema: JSON Schema dict or Pydantic model class

    Returns:
        ConstraintNode: Root of the constraint tree

    Raises:
        ValueError: If schema is invalid or unsupported

    Example:
        ```python
        # JSON Schema
        schema = {"type": "string", "minLength": 3}
        constraint = parse_schema(schema)

        # Pydantic model
        from pydantic import BaseModel
        class User(BaseModel):
            name: str
            age: int

        constraint = parse_schema(User)
        ```
    """
    # Check if it's a Pydantic model
    if isinstance(schema, type):
        try:
            # Import here to avoid circular dependency
            from grammar_guard.schema.pydantic_adapter import pydantic_to_schema
            schema = pydantic_to_schema(schema)
        except ImportError:
            raise ValueError("Schema appears to be a class but Pydantic is not available")

    # Now parse as JSON Schema dict
    return _parse_schema_dict(schema)


def _parse_schema_dict(schema: Dict[str, Any]) -> ConstraintNode:
    """
    Internal method to parse a JSON Schema dictionary.

    Args:
        schema: JSON Schema dictionary

    Returns:
        ConstraintNode: Parsed constraint

    Raises:
        ValueError: If schema type is unsupported
    """
    # Handle $ref (not supported in MVP)
    if "$ref" in schema:
        raise ValueError("$ref is not supported in MVP - use inline schemas")

    # Handle anyOf, oneOf
    if "anyOf" in schema:
        return _parse_union(schema["anyOf"])
    if "oneOf" in schema:
        # Treat oneOf same as anyOf for MVP
        return _parse_union(schema["oneOf"])

    # Get type - may be string or list of strings
    schema_type = schema.get("type")

    if schema_type is None:
        # No type specified - try to infer from properties
        if "properties" in schema:
            schema_type = "object"
        elif "items" in schema:
            schema_type = "array"
        else:
            # Default to string
            schema_type = "string"

    # Handle array of types (e.g., ["string", "null"])
    if isinstance(schema_type, list):
        return _parse_union([{"type": t, **{k: v for k, v in schema.items() if k != "type"}}
                            for t in schema_type])

    # Dispatch based on type
    if schema_type == "object":
        return _parse_object(schema)
    elif schema_type == "array":
        return _parse_array(schema)
    elif schema_type == "string":
        return _parse_string(schema)
    elif schema_type == "integer":
        return _parse_number(schema, is_integer=True)
    elif schema_type == "number":
        return _parse_number(schema, is_integer=False)
    elif schema_type == "boolean":
        return _parse_boolean(schema)
    elif schema_type == "null":
        return _parse_null(schema)
    else:
        raise ValueError(f"Unsupported schema type: {schema_type}")


def _parse_object(schema: Dict[str, Any]) -> ObjectConstraint:
    """
    Parse an object schema into ObjectConstraint.

    Args:
        schema: JSON Schema dict with type=object

    Returns:
        ObjectConstraint: Parsed object constraint
    """
    properties = {}
    for prop_name, prop_schema in schema.get("properties", {}).items():
        properties[prop_name] = _parse_schema_dict(prop_schema)

    required = set(schema.get("required", []))
    additional_properties = schema.get("additionalProperties", False)

    return ObjectConstraint(
        properties=properties,
        required=required,
        additional_properties=additional_properties
    )


def _parse_array(schema: Dict[str, Any]) -> ArrayConstraint:
    """
    Parse an array schema into ArrayConstraint.

    Args:
        schema: JSON Schema dict with type=array

    Returns:
        ArrayConstraint: Parsed array constraint
    """
    items_schema = schema.get("items")
    items_constraint = _parse_schema_dict(items_schema) if items_schema else None

    min_items = schema.get("minItems")
    max_items = schema.get("maxItems")

    return ArrayConstraint(
        items=items_constraint,
        min_items=min_items,
        max_items=max_items
    )


def _parse_string(schema: Dict[str, Any]) -> StringConstraint:
    """
    Parse a string schema into StringConstraint.

    Args:
        schema: JSON Schema dict with type=string

    Returns:
        StringConstraint: Parsed string constraint
    """
    min_length = schema.get("minLength")
    max_length = schema.get("maxLength")
    pattern = schema.get("pattern")

    # Handle enum
    enum_values = schema.get("enum")
    enum_set = set(enum_values) if enum_values else None

    # Handle const (single allowed value)
    const_value = schema.get("const")
    if const_value is not None:
        enum_set = {const_value}

    return StringConstraint(
        min_length=min_length,
        max_length=max_length,
        pattern=pattern,
        enum=enum_set
    )


def _parse_number(schema: Dict[str, Any], is_integer: bool) -> NumberConstraint:
    """
    Parse a number or integer schema into NumberConstraint.

    Args:
        schema: JSON Schema dict with type=number or type=integer
        is_integer: True for integer, False for number

    Returns:
        NumberConstraint: Parsed number constraint
    """
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")

    # Handle enum
    enum_values = schema.get("enum")
    enum_set = set(enum_values) if enum_values else None

    # Handle const
    const_value = schema.get("const")
    if const_value is not None:
        enum_set = {const_value}

    # Handle exclusiveMinimum/exclusiveMaximum
    if "exclusiveMinimum" in schema:
        exclusive_min = schema["exclusiveMinimum"]
        if is_integer:
            minimum = exclusive_min + 1
        else:
            # For floats, we can't easily represent exclusive bounds in our model
            # so we'll use the bound itself and rely on post-validation
            minimum = exclusive_min

    if "exclusiveMaximum" in schema:
        exclusive_max = schema["exclusiveMaximum"]
        if is_integer:
            maximum = exclusive_max - 1
        else:
            maximum = exclusive_max

    return NumberConstraint(
        is_integer=is_integer,
        minimum=minimum,
        maximum=maximum,
        enum=enum_set
    )


def _parse_boolean(schema: Dict[str, Any]) -> BooleanConstraint:
    """
    Parse a boolean schema into BooleanConstraint.

    Args:
        schema: JSON Schema dict with type=boolean

    Returns:
        BooleanConstraint: Parsed boolean constraint
    """
    return BooleanConstraint()


def _parse_null(schema: Dict[str, Any]) -> NullConstraint:
    """
    Parse a null schema into NullConstraint.

    Args:
        schema: JSON Schema dict with type=null

    Returns:
        NullConstraint: Parsed null constraint
    """
    return NullConstraint()


def _parse_union(schemas: List[Dict[str, Any]]) -> UnionConstraint:
    """
    Parse a union (anyOf/oneOf) into UnionConstraint.

    Args:
        schemas: List of JSON Schema dicts

    Returns:
        UnionConstraint: Parsed union constraint
    """
    options = [_parse_schema_dict(s) for s in schemas]
    return UnionConstraint(options=options)


def normalize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a JSON Schema for easier processing.

    This function:
    - Adds default values for missing fields
    - Converts shorthand forms to canonical forms
    - Validates schema structure

    Args:
        schema: JSON Schema dictionary

    Returns:
        Dict: Normalized schema

    Example:
        ```python
        # Add default values
        schema = {"type": "object"}
        normalized = normalize_schema(schema)
        # normalized = {"type": "object", "properties": {}, "required": []}
        ```
    """
    normalized = schema.copy()

    # Add defaults based on type
    schema_type = normalized.get("type")

    if schema_type == "object":
        normalized.setdefault("properties", {})
        normalized.setdefault("required", [])
        normalized.setdefault("additionalProperties", False)

    elif schema_type == "array":
        if "items" not in normalized:
            # Default to any type
            normalized["items"] = {}

    elif schema_type == "string":
        # No defaults needed
        pass

    return normalized


def validate_schema(schema: Dict[str, Any]) -> None:
    """
    Validate that a schema is well-formed and supported.

    Args:
        schema: JSON Schema dictionary

    Raises:
        ValueError: If schema is invalid or uses unsupported features

    Example:
        ```python
        schema = {"type": "unknown"}
        validate_schema(schema)  # Raises ValueError
        ```
    """
    # Check for unsupported features
    unsupported_keywords = ["$ref", "allOf", "not", "if", "then", "else"]
    for keyword in unsupported_keywords:
        if keyword in schema:
            raise ValueError(f"Keyword '{keyword}' is not supported in MVP")

    # Check type is valid
    valid_types = ["object", "array", "string", "integer", "number", "boolean", "null"]
    schema_type = schema.get("type")

    if schema_type is not None:
        if isinstance(schema_type, str):
            if schema_type not in valid_types:
                raise ValueError(f"Invalid type: {schema_type}")
        elif isinstance(schema_type, list):
            for t in schema_type:
                if t not in valid_types:
                    raise ValueError(f"Invalid type in type array: {t}")
        else:
            raise ValueError(f"Type must be string or array, got: {type(schema_type)}")

    # Validate nested schemas
    if "properties" in schema:
        for prop_schema in schema["properties"].values():
            validate_schema(prop_schema)

    if "items" in schema:
        validate_schema(schema["items"])

    if "anyOf" in schema:
        for sub_schema in schema["anyOf"]:
            validate_schema(sub_schema)

    if "oneOf" in schema:
        for sub_schema in schema["oneOf"]:
            validate_schema(sub_schema)

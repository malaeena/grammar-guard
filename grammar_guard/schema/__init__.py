"""
Schema parsing and processing module.

This module handles conversion of JSON Schema and Pydantic models into internal
constraint representations, compilation to regex patterns, and schema simplification
for retry logic.

Components:
    - types: Constraint type definitions (ObjectConstraint, StringConstraint, etc.)
    - parser: Main entry point for schema parsing
    - pydantic_adapter: Convert Pydantic models to JSON Schema
    - regex_compiler: Convert constraints to regex patterns for FSM building
    - simplifier: Progressive schema simplification for retry attempts

Example:
    ```python
    from grammar_guard.schema import parse_schema
    from pydantic import BaseModel

    class User(BaseModel):
        name: str
        age: int

    # Parse Pydantic model
    constraint = parse_schema(User)

    # Or parse JSON Schema dict
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    constraint = parse_schema(schema)
    ```
"""

from grammar_guard.schema.parser import parse_schema, normalize_schema, validate_schema
from grammar_guard.schema.pydantic_adapter import pydantic_to_schema, is_pydantic_model
from grammar_guard.schema.regex_compiler import compile_to_regex
from grammar_guard.schema.simplifier import simplify_schema, get_simplification_summary

__all__ = [
    "parse_schema",
    "normalize_schema",
    "validate_schema",
    "pydantic_to_schema",
    "is_pydantic_model",
    "compile_to_regex",
    "simplify_schema",
    "get_simplification_summary",
]

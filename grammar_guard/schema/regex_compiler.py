"""
Regex compiler - convert constraint trees to regex patterns for FSM building.

This module provides the interface for converting parsed schemas (represented as
ConstraintNode trees) into regular expression patterns. These regex patterns are
then used to build character-level FSMs via the interegular library.

The actual regex generation logic is implemented in each ConstraintNode's to_regex()
method (see types.py). This module provides convenience functions and optimization.

Usage:
    ```python
    from grammar_guard.schema import parse_schema
    from grammar_guard.schema.regex_compiler import compile_to_regex

    schema = {"type": "string", "minLength": 3, "maxLength": 10}
    constraint = parse_schema(schema)

    regex = compile_to_regex(constraint)
    # regex = '"[^"]{3,10}"'
    ```

Regex Strategy:
    - Objects: \{"key1":"value1","key2":"value2"\}
    - Arrays: \["item1","item2"\]
    - Strings: "[^"]{min,max}" (handles length constraints)
    - Numbers: -?(0|[1-9][0-9]*) for integers
    - Booleans: (true|false)
    - Null: null
    - Unions: (option1|option2|option3)

Limitations:
    - Doesn't handle escaped quotes in strings (simplified for MVP)
    - Number range constraints validated post-generation (hard to express in regex)
    - Optional object properties use simplified pattern generation
"""

import re
from typing import Optional

from grammar_guard.schema.types import ConstraintNode


def compile_to_regex(constraint: ConstraintNode, optimize: bool = True) -> str:
    """
    Compile a constraint tree to a regular expression pattern.

    Args:
        constraint: Root of the constraint tree
        optimize: If True, apply regex optimizations (default: True)

    Returns:
        str: Regular expression pattern matching the constraint

    Example:
        ```python
        from grammar_guard.schema import parse_schema
        from grammar_guard.schema.regex_compiler import compile_to_regex

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }

        constraint = parse_schema(schema)
        regex = compile_to_regex(constraint)
        print(regex)
        # \{"name"\s*:\s*"[^"]*"\s*,\s*"age"\s*:\s*-?(0|[1-9][0-9]*)\s*\}
        ```
    """
    # Get regex from constraint
    regex = constraint.to_regex()

    # Apply optimizations if requested
    if optimize:
        regex = optimize_regex(regex)

    return regex


def optimize_regex(regex: str) -> str:
    """
    Apply optimizations to make regex more efficient.

    Optimizations:
    - Reduce redundant whitespace patterns: \\s*\\s* → \\s*
    - Simplify nested groups: ((a)) → (a)
    - Remove unnecessary non-capturing groups in simple cases

    Args:
        regex: Input regex pattern

    Returns:
        str: Optimized regex pattern

    Note:
        These optimizations are conservative to avoid breaking correctness.
        They primarily reduce pattern length and improve readability.
    """
    # Reduce redundant whitespace patterns
    regex = re.sub(r'\\s\*\\s\*', r'\\s*', regex)
    regex = re.sub(r'\\s\+\\s\+', r'\\s+', regex)

    # Simplify nested groups (conservative - only non-capturing groups)
    # This is tricky to do safely, so we'll keep it simple for MVP
    # regex = re.sub(r'\(\(([^()]+)\)\)', r'(\1)', regex)

    return regex


def validate_regex(regex: str) -> bool:
    """
    Validate that a regex pattern is well-formed.

    Args:
        regex: Regular expression pattern to validate

    Returns:
        bool: True if regex is valid, False otherwise

    Example:
        ```python
        assert validate_regex(r'"[^"]*"') == True
        assert validate_regex(r'"[^"*"') == False  # Unclosed bracket
        ```
    """
    try:
        re.compile(regex)
        return True
    except re.error:
        return False


def get_regex_complexity(regex: str) -> int:
    """
    Estimate the complexity of a regex pattern.

    This is a simple heuristic based on pattern length and features used.
    Used for deciding whether to cache compiled patterns.

    Args:
        regex: Regular expression pattern

    Returns:
        int: Complexity score (higher = more complex)

    Complexity factors:
    - Base: pattern length
    - +10 for each alternation (|)
    - +20 for each quantifier ({m,n})
    - +30 for each lookahead/lookbehind

    Example:
        ```python
        simple = '"[^"]*"'
        complex = '\\{"name":"[^"]{3,50}","age":(0|[1-9][0-9]*)\\}'

        assert get_regex_complexity(complex) > get_regex_complexity(simple)
        ```
    """
    complexity = len(regex)

    # Count features
    complexity += regex.count('|') * 10  # Alternation
    complexity += regex.count('{') * 20  # Quantifiers
    complexity += regex.count('(?=') * 30  # Lookahead
    complexity += regex.count('(?!') * 30  # Negative lookahead
    complexity += regex.count('(?<=') * 30  # Lookbehind
    complexity += regex.count('(?<!') * 30  # Negative lookbehind

    return complexity


def explain_regex(regex: str, indent: int = 0) -> str:
    """
    Generate human-readable explanation of a regex pattern.

    This is useful for debugging and understanding what constraints
    are actually being enforced.

    Args:
        regex: Regular expression pattern
        indent: Indentation level for nested explanations

    Returns:
        str: Human-readable explanation

    Example:
        ```python
        regex = r'"[^"]{3,10}"'
        explanation = explain_regex(regex)
        print(explanation)
        # String with:
        #   - Minimum length: 3
        #   - Maximum length: 10
        ```

    Note:
        This is a simplified explanation - full regex parsing is complex.
        We provide basic pattern recognition for common cases.
    """
    indent_str = "  " * indent
    explanations = []

    # Detect common patterns
    if regex.startswith('\\{') and regex.endswith('\\}'):
        explanations.append(f"{indent_str}JSON Object")

    elif regex.startswith('\\[') and regex.endswith('\\]'):
        explanations.append(f"{indent_str}JSON Array")

    elif regex.startswith('"') and regex.endswith('"'):
        # String pattern
        if '{' in regex:
            # Extract quantifier
            match = re.search(r'\{(\d+),(\d+)\}', regex)
            if match:
                min_len, max_len = match.groups()
                explanations.append(f"{indent_str}String with length {min_len}-{max_len}")
            else:
                match = re.search(r'\{(\d+),\}', regex)
                if match:
                    min_len = match.group(1)
                    explanations.append(f"{indent_str}String with minimum length {min_len}")
                else:
                    explanations.append(f"{indent_str}String")
        else:
            explanations.append(f"{indent_str}String (any length)")

    elif regex in ['(true|false)', 'true|false']:
        explanations.append(f"{indent_str}Boolean")

    elif regex == 'null':
        explanations.append(f"{indent_str}Null")

    elif re.match(r'-?\(0\|\[1-9\]\[0-9\]\*\)', regex):
        explanations.append(f"{indent_str}Integer")

    elif '|' in regex:
        options = regex.strip('()').split('|')
        explanations.append(f"{indent_str}One of:")
        for opt in options[:5]:  # Limit to first 5 for readability
            explanations.append(f"{indent_str}  - {opt}")
        if len(options) > 5:
            explanations.append(f"{indent_str}  ... and {len(options) - 5} more")

    else:
        # Generic
        explanations.append(f"{indent_str}Pattern: {regex[:50]}{'...' if len(regex) > 50 else ''}")

    return '\n'.join(explanations)


def test_regex_on_examples(regex: str, examples: list) -> dict:
    """
    Test a regex pattern against example strings.

    Useful for validating that generated regex actually matches intended inputs.

    Args:
        regex: Regular expression pattern
        examples: List of example strings to test

    Returns:
        dict: Results with keys 'passed', 'failed', 'errors'

    Example:
        ```python
        regex = r'"[^"]{3,10}"'
        examples = ['"hello"', '"hi"', '"too long string"']

        results = test_regex_on_examples(regex, examples)
        # results = {
        #     'passed': ['"hello"'],
        #     'failed': ['"hi"', '"too long string"'],
        #     'errors': []
        # }
        ```
    """
    try:
        pattern = re.compile(regex)
    except re.error as e:
        return {
            'passed': [],
            'failed': [],
            'errors': [f"Invalid regex: {e}"]
        }

    passed = []
    failed = []

    for example in examples:
        match = pattern.fullmatch(example)
        if match:
            passed.append(example)
        else:
            failed.append(example)

    return {
        'passed': passed,
        'failed': failed,
        'errors': []
    }

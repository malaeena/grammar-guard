"""
Constraint type definitions for JSON Schema representation.

This module defines the internal constraint type hierarchy used to represent
JSON schemas in a format suitable for regex compilation and FSM building.

Type Hierarchy:
    ConstraintNode (abstract)
    ├── ObjectConstraint: Represents JSON objects with properties and required fields
    ├── ArrayConstraint: Represents JSON arrays with item constraints
    ├── StringConstraint: Represents strings with length and enum constraints
    ├── NumberConstraint: Represents numbers (int/float) with range constraints
    ├── BooleanConstraint: Represents boolean values
    ├── NullConstraint: Represents null values
    └── UnionConstraint: Represents multiple possible types (anyOf)

Each constraint type knows how to:
    - Validate its configuration
    - Convert itself to a regex pattern
    - Simplify itself for retry logic
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union


@dataclass
class ConstraintNode(ABC):
    """
    Abstract base class for all constraint types.

    All constraints can be converted to regex patterns and simplified for retry logic.
    """

    @abstractmethod
    def to_regex(self) -> str:
        """
        Convert this constraint to a regex pattern.

        Returns:
            str: Regular expression pattern matching this constraint

        Note:
            The regex should match the complete JSON representation of this type,
            including quotes for strings, braces for objects, etc.
        """
        pass

    @abstractmethod
    def simplify(self, level: int) -> "ConstraintNode":
        """
        Create a simplified version of this constraint for retry logic.

        Args:
            level: Simplification level (1-5, higher = more simplified)
                1: Remove optional fields
                2: Relax length constraints by 50%
                3: Relax numeric constraints by 2x
                4: Remove enum constraints
                5: Convert to most permissive form

        Returns:
            ConstraintNode: A simplified version of this constraint

        Note:
            This is used when generation fails - we progressively relax
            constraints to give the model more freedom.
        """
        pass


@dataclass
class ObjectConstraint(ConstraintNode):
    """
    Represents a JSON object with typed properties.

    Example JSON Schema:
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 3},
                "age": {"type": "integer", "minimum": 0}
            },
            "required": ["name"]
        }

    Attributes:
        properties: Dict mapping property names to their constraints
        required: Set of required property names
        additional_properties: Whether to allow properties not in schema
    """

    properties: Dict[str, ConstraintNode] = field(default_factory=dict)
    required: Set[str] = field(default_factory=set)
    additional_properties: bool = False

    def to_regex(self) -> str:
        """
        Generate regex for object like: {"name":"value","age":123}

        Strategy:
            - Start with opening brace: \{
            - For each property: "key"\s*:\s*<value_regex>
            - Separate properties with commas and optional whitespace
            - End with closing brace: \}

        Returns:
            str: Regex pattern matching this object structure
        """
        if not self.properties:
            # Empty object
            return r'\{\s*\}'

        # Build property patterns
        property_patterns = []
        for key, value_constraint in self.properties.items():
            # Property format: "key": <value>
            key_pattern = f'"{key}"'
            value_pattern = value_constraint.to_regex()
            property_patterns.append(f'{key_pattern}\\s*:\\s*{value_pattern}')

        # Required properties must appear, optional properties may appear
        required_props = [p for k, p in zip(self.properties.keys(), property_patterns) if k in self.required]
        optional_props = [p for k, p in zip(self.properties.keys(), property_patterns) if k not in self.required]

        if required_props and not optional_props:
            # All required: simple join with commas
            inner = ',\\s*'.join(required_props)
        elif required_props and optional_props:
            # Mix of required and optional: required must appear, optional may appear
            # For simplicity in MVP, we'll generate pattern with all properties
            # and use | operator for variations
            inner = ',\\s*'.join(required_props + [f'({p})?' for p in optional_props])
        else:
            # All optional
            inner = ',\\s*'.join([f'({p})?' for p in optional_props])

        return f'\\{{\\s*{inner}\\s*\\}}'

    def simplify(self, level: int) -> "ObjectConstraint":
        """
        Simplify object constraint based on level.

        Level 1: Remove all optional fields (keep only required)
        Level 2+: Simplify all property constraints

        Args:
            level: Simplification level

        Returns:
            ObjectConstraint: Simplified object constraint
        """
        if level >= 1:
            # Remove optional properties
            simplified_props = {
                k: v.simplify(level) for k, v in self.properties.items() if k in self.required
            }
            return ObjectConstraint(
                properties=simplified_props,
                required=self.required.copy(),
                additional_properties=self.additional_properties
            )
        else:
            # Just simplify properties
            return ObjectConstraint(
                properties={k: v.simplify(level) for k, v in self.properties.items()},
                required=self.required.copy(),
                additional_properties=self.additional_properties
            )


@dataclass
class ArrayConstraint(ConstraintNode):
    """
    Represents a JSON array with item constraints.

    Example JSON Schema:
        {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 10
        }

    Attributes:
        items: Constraint for array items (all items must match)
        min_items: Minimum number of items (None = no limit)
        max_items: Maximum number of items (None = no limit)
    """

    items: Optional[ConstraintNode] = None
    min_items: Optional[int] = None
    max_items: Optional[int] = None

    def to_regex(self) -> str:
        """
        Generate regex for array like: ["item1","item2","item3"]

        Strategy:
            - Start with opening bracket: \[
            - Repeat item pattern with comma separators
            - Use {min,max} quantifier for repetition
            - End with closing bracket: \]

        Returns:
            str: Regex pattern matching this array structure
        """
        if self.items is None:
            # Empty array
            return r'\[\s*\]'

        item_pattern = self.items.to_regex()

        # Build repetition pattern
        if self.min_items is None and self.max_items is None:
            # Any number of items: item(,item)*
            repetition = f'({item_pattern}(\\s*,\\s*{item_pattern})*)?'
        elif self.min_items is not None and self.max_items is not None:
            # Fixed range: need exactly min to max items
            if self.min_items == 0:
                if self.max_items == 0:
                    return r'\[\s*\]'
                else:
                    # 0 to max items
                    repetition = f'({item_pattern}(\\s*,\\s*{item_pattern}){{0,{self.max_items-1}}})?'
            else:
                # min to max items (min >= 1)
                repetition = f'{item_pattern}(\\s*,\\s*{item_pattern}){{{self.min_items-1},{self.max_items-1}}}'
        elif self.min_items is not None:
            # At least min items
            if self.min_items == 0:
                repetition = f'({item_pattern}(\\s*,\\s*{item_pattern})*)?'
            else:
                repetition = f'{item_pattern}(\\s*,\\s*{item_pattern}){{{self.min_items-1},}}'
        else:
            # At most max items
            repetition = f'({item_pattern}(\\s*,\\s*{item_pattern}){{0,{self.max_items-1}}})?'

        return f'\\[\\s*{repetition}\\s*\\]'

    def simplify(self, level: int) -> "ArrayConstraint":
        """
        Simplify array constraint based on level.

        Level 2: Relax length constraints by 50%
        Level 3+: Remove all length constraints

        Args:
            level: Simplification level

        Returns:
            ArrayConstraint: Simplified array constraint
        """
        simplified_items = self.items.simplify(level) if self.items else None

        if level >= 3:
            # Remove all length constraints
            return ArrayConstraint(items=simplified_items, min_items=None, max_items=None)
        elif level >= 2:
            # Relax by 50%
            min_items = max(0, int(self.min_items * 0.5)) if self.min_items is not None else None
            max_items = int(self.max_items * 1.5) if self.max_items is not None else None
            return ArrayConstraint(items=simplified_items, min_items=min_items, max_items=max_items)
        else:
            return ArrayConstraint(
                items=simplified_items,
                min_items=self.min_items,
                max_items=self.max_items
            )


@dataclass
class StringConstraint(ConstraintNode):
    """
    Represents a JSON string with validation constraints.

    Example JSON Schema:
        {
            "type": "string",
            "minLength": 3,
            "maxLength": 50,
            "enum": ["red", "green", "blue"]
        }

    Attributes:
        min_length: Minimum string length (None = no limit)
        max_length: Maximum string length (None = no limit)
        pattern: Regex pattern string must match (None = any string)
        enum: Set of allowed values (None = any string)
    """

    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    enum: Optional[Set[str]] = None

    def to_regex(self) -> str:
        """
        Generate regex for JSON string like: "value"

        Strategy:
            - Always include quotes: "..."
            - Use character class [^"] to match any non-quote character
            - Use {min,max} quantifier for length constraints
            - Use | operator for enum values

        Returns:
            str: Regex pattern matching this string constraint

        Note:
            For simplicity, we don't handle escaped quotes in MVP.
            This means strings like "hello \"world\"" won't be properly constrained.
        """
        if self.enum:
            # Enum: exact matches only
            enum_values = '|'.join(self.enum)
            return f'"({enum_values})"'

        # Build character pattern
        if self.pattern:
            # Use provided pattern (between quotes)
            return f'"{self.pattern}"'
        else:
            # Default: any non-quote characters
            char_pattern = '[^"\\\\]'  # No quotes or backslashes for simplicity

            # Apply length constraints
            if self.min_length is None and self.max_length is None:
                # Any length: [^"]*
                quantifier = '*'
            elif self.min_length is not None and self.max_length is not None:
                # Fixed range: {min,max}
                quantifier = f'{{{self.min_length},{self.max_length}}}'
            elif self.min_length is not None:
                # At least min: {min,}
                quantifier = f'{{{self.min_length},}}'
            else:
                # At most max: {0,max}
                quantifier = f'{{0,{self.max_length}}}'

            return f'"{char_pattern}{quantifier}"'

    def simplify(self, level: int) -> "StringConstraint":
        """
        Simplify string constraint based on level.

        Level 2: Relax length constraints by 50%
        Level 3: Remove length constraints
        Level 4: Remove enum and pattern constraints

        Args:
            level: Simplification level

        Returns:
            StringConstraint: Simplified string constraint
        """
        if level >= 4:
            # Remove all constraints
            return StringConstraint()
        elif level >= 3:
            # Remove length constraints, keep enum/pattern
            return StringConstraint(pattern=self.pattern, enum=self.enum)
        elif level >= 2:
            # Relax length by 50%
            min_len = max(0, int(self.min_length * 0.5)) if self.min_length is not None else None
            max_len = int(self.max_length * 1.5) if self.max_length is not None else None
            return StringConstraint(
                min_length=min_len,
                max_length=max_len,
                pattern=self.pattern,
                enum=self.enum
            )
        else:
            return StringConstraint(
                min_length=self.min_length,
                max_length=self.max_length,
                pattern=self.pattern,
                enum=self.enum.copy() if self.enum else None
            )


@dataclass
class NumberConstraint(ConstraintNode):
    """
    Represents a JSON number (integer or float) with range constraints.

    Example JSON Schema:
        {
            "type": "integer",
            "minimum": 0,
            "maximum": 100
        }

    Attributes:
        is_integer: True for integer, False for float
        minimum: Minimum value (None = no limit)
        maximum: Maximum value (None = no limit)
        enum: Set of allowed values (None = any number)
    """

    is_integer: bool = True
    minimum: Optional[Union[int, float]] = None
    maximum: Optional[Union[int, float]] = None
    enum: Optional[Set[Union[int, float]]] = None

    def to_regex(self) -> str:
        """
        Generate regex for JSON number like: 123 or 45.67

        Strategy:
            - For integers: -?[0-9]+
            - For floats: -?[0-9]+(\.[0-9]+)?
            - Enum: exact value matches with | operator

        Returns:
            str: Regex pattern matching this number constraint

        Note:
            Range constraints (minimum/maximum) are difficult to enforce with regex.
            We validate these in post-generation validation instead.
        """
        if self.enum:
            # Enum: exact matches only
            enum_values = '|'.join(str(v) for v in self.enum)
            return f'({enum_values})'

        if self.is_integer:
            # Integer pattern: optional minus, then digits
            return r'-?(0|[1-9][0-9]*)'
        else:
            # Float pattern: optional minus, digits, optional decimal part
            return r'-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?'

    def simplify(self, level: int) -> "NumberConstraint":
        """
        Simplify number constraint based on level.

        Level 3: Relax range constraints by 2x
        Level 4: Remove enum constraints
        Level 5: Remove all range constraints

        Args:
            level: Simplification level

        Returns:
            NumberConstraint: Simplified number constraint
        """
        if level >= 5:
            # Remove all constraints
            return NumberConstraint(is_integer=self.is_integer)
        elif level >= 4:
            # Remove enum, keep range
            return NumberConstraint(
                is_integer=self.is_integer,
                minimum=self.minimum,
                maximum=self.maximum
            )
        elif level >= 3:
            # Relax range by 2x
            if self.minimum is not None and self.maximum is not None:
                center = (self.minimum + self.maximum) / 2
                range_size = self.maximum - self.minimum
                new_min = center - range_size
                new_max = center + range_size
            elif self.minimum is not None:
                new_min = self.minimum * 0.5 if self.minimum > 0 else self.minimum * 2
                new_max = None
            elif self.maximum is not None:
                new_min = None
                new_max = self.maximum * 1.5 if self.maximum > 0 else self.maximum * 0.5
            else:
                new_min = None
                new_max = None

            return NumberConstraint(
                is_integer=self.is_integer,
                minimum=new_min,
                maximum=new_max,
                enum=self.enum.copy() if self.enum else None
            )
        else:
            return NumberConstraint(
                is_integer=self.is_integer,
                minimum=self.minimum,
                maximum=self.maximum,
                enum=self.enum.copy() if self.enum else None
            )


@dataclass
class BooleanConstraint(ConstraintNode):
    """
    Represents a JSON boolean value.

    Example JSON Schema:
        {"type": "boolean"}

    No additional constraints - just true or false.
    """

    def to_regex(self) -> str:
        """Generate regex for boolean: true or false"""
        return r'(true|false)'

    def simplify(self, level: int) -> "BooleanConstraint":
        """Booleans cannot be simplified"""
        return BooleanConstraint()


@dataclass
class NullConstraint(ConstraintNode):
    """
    Represents a JSON null value.

    Example JSON Schema:
        {"type": "null"}
    """

    def to_regex(self) -> str:
        """Generate regex for null"""
        return r'null'

    def simplify(self, level: int) -> "NullConstraint":
        """Null cannot be simplified"""
        return NullConstraint()


@dataclass
class UnionConstraint(ConstraintNode):
    """
    Represents a union of multiple possible types (anyOf, oneOf).

    Example JSON Schema:
        {
            "anyOf": [
                {"type": "string"},
                {"type": "integer"}
            ]
        }

    Attributes:
        options: List of possible constraint types
    """

    options: List[ConstraintNode] = field(default_factory=list)

    def to_regex(self) -> str:
        """
        Generate regex for union: (option1|option2|option3)

        Strategy:
            - Convert each option to regex
            - Join with | operator
            - Wrap in parentheses for grouping

        Returns:
            str: Regex pattern matching any of the union options
        """
        if not self.options:
            # Empty union - shouldn't happen, but handle gracefully
            return r'null'

        if len(self.options) == 1:
            # Single option - no need for union
            return self.options[0].to_regex()

        # Multiple options: join with |
        option_patterns = [opt.to_regex() for opt in self.options]
        return '(' + '|'.join(option_patterns) + ')'

    def simplify(self, level: int) -> "UnionConstraint":
        """
        Simplify union constraint by simplifying all options.

        Args:
            level: Simplification level

        Returns:
            UnionConstraint: Union with all options simplified
        """
        return UnionConstraint(options=[opt.simplify(level) for opt in self.options])

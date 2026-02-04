#!/usr/bin/env python3
"""
Demo: Person record with nested fields.

This demonstrates generating a person profile with:
- Required fields: name, age
- Nested object: address with city (required)
- Array: hobbies
- Validation constraints: minLength, maxLength, minimum, maximum

From AGENTS.md spec requirement #1.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from grammar_guard import GrammarConstrainedGenerator


def main():
    print("=" * 60)
    print("GrammarGuard Demo: Person Record with Nested Fields")
    print("=" * 60)

    # Define schema
    schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "minLength": 2,
                "maxLength": 50
            },
            "age": {
                "type": "integer",
                "minimum": 0,
                "maximum": 150
            },
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                    "zipcode": {
                        "type": "string",
                        "minLength": 5,
                        "maxLength": 10
                    }
                },
                "required": ["city"]
            },
            "hobbies": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
                "maxItems": 10
            }
        },
        "required": ["name", "age"]
    }

    print("\nSchema:")
    print(json.dumps(schema, indent=2))

    # Initialize generator
    print("\n" + "=" * 60)
    print("Initializing Generator...")
    print("=" * 60)

    generator = GrammarConstrainedGenerator(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        backend="transformers",
        device=None  # Auto-detect
    )

    print("✓ Generator initialized")

    # Test prompts
    prompts = [
        "Generate a person named Alice, age 28, living in NYC with hobbies reading and hiking",
        "Create a user profile for Bob Smith, 35 years old, residing in San Francisco",
        "Make a person record for Charlie, age 42"
    ]

    # Generate for each prompt
    for i, prompt in enumerate(prompts, 1):
        print("\n" + "=" * 60)
        print(f"Test {i}/3")
        print("=" * 60)
        print(f"Prompt: {prompt}")

        result = generator.generate(
            prompt=prompt,
            schema=schema,
            max_tokens=200,
            max_retries=3
        )

        print(f"\n{'='*30}")
        print(f"Result:")
        print(f"{'='*30}")
        print(f"Valid: {'✓' if result.is_valid else '✗'} {result.is_valid}")
        print(f"Retries: {result.retries}")
        print(f"Simplification Level: {result.simplification_level}")
        print(f"Latency: {result.latency_ms:.0f}ms")
        print(f"Tokens: {result.tokens_generated}")

        print(f"\nOutput:")
        if result.is_valid:
            # Pretty print JSON
            try:
                parsed = json.loads(result.output)
                print(json.dumps(parsed, indent=2))

                # Show field details
                print(f"\nField Analysis:")
                print(f"  Name: {parsed.get('name', 'N/A')} (length: {len(parsed.get('name', ''))})")
                print(f"  Age: {parsed.get('age', 'N/A')}")

                if "address" in parsed:
                    print(f"  Address:")
                    for key, val in parsed["address"].items():
                        print(f"    {key}: {val}")

                if "hobbies" in parsed:
                    print(f"  Hobbies: {', '.join(parsed['hobbies'])}")

            except json.JSONDecodeError:
                print(result.output)
        else:
            print(result.output)
            print(f"\nValidation Errors:")
            for error in result.validation_errors:
                print(f"  - {error}")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

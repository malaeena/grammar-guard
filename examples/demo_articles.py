#!/usr/bin/env python3
"""
Demo: List of articles with authors and timestamps.

This demonstrates generating an array of articles with:
- Array of objects
- Required fields per article: title, author, timestamp
- Optional fields: tags array
- Validation constraints: minLength, maxLength, minItems, maxItems

From AGENTS.md spec requirement #2.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from grammar_guard import GrammarConstrainedGenerator


def main():
    print("=" * 60)
    print("GrammarGuard Demo: List of Articles")
    print("=" * 60)

    # Define schema
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "minLength": 5,
                    "maxLength": 200
                },
                "author": {
                    "type": "string",
                    "minLength": 2,
                    "maxLength": 100
                },
                "timestamp": {
                    "type": "string"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["title", "author", "timestamp"]
        },
        "minItems": 1,
        "maxItems": 10
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
        device=None
    )

    print("✓ Generator initialized")

    # Test prompts
    prompts = [
        "Generate 3 technology articles with authors and timestamps",
        "Create a list of 2 news articles about artificial intelligence",
    ]

    # Generate for each prompt
    for i, prompt in enumerate(prompts, 1):
        print("\n" + "=" * 60)
        print(f"Test {i}/{len(prompts)}")
        print("=" * 60)
        print(f"Prompt: {prompt}")

        result = generator.generate(
            prompt=prompt,
            schema=schema,
            max_tokens=300,
            max_retries=3
        )

        print(f"\n{'='*30}")
        print(f"Result:")
        print(f"{'='*30}")
        print(f"Valid: {'✓' if result.is_valid else '✗'} {result.is_valid}")
        print(f"Retries: {result.retries}")
        print(f"Latency: {result.latency_ms:.0f}ms")
        print(f"Tokens: {result.tokens_generated}")

        print(f"\nOutput:")
        if result.is_valid:
            try:
                articles = json.loads(result.output)
                print(f"Generated {len(articles)} article(s):\n")

                for idx, article in enumerate(articles, 1):
                    print(f"Article {idx}:")
                    print(f"  Title: {article.get('title', 'N/A')}")
                    print(f"  Author: {article.get('author', 'N/A')}")
                    print(f"  Timestamp: {article.get('timestamp', 'N/A')}")

                    if "tags" in article and article["tags"]:
                        print(f"  Tags: {', '.join(article['tags'])}")

                    print()

                # Full JSON
                print("Full JSON:")
                print(json.dumps(articles, indent=2))

            except json.JSONDecodeError:
                print(result.output)
        else:
            print(result.output)

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

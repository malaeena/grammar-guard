#!/usr/bin/env python3
"""
Demo: JSON-RPC function call with arguments.

This demonstrates generating JSON-RPC 2.0 function calls with:
- Const constraint: jsonrpc must be "2.0"
- Required fields: jsonrpc, method, id
- Optional fields: params object
- Type validation: id must be integer

From AGENTS.md spec requirement #3.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from grammar_guard import GrammarConstrainedGenerator


def main():
    print("=" * 60)
    print("GrammarGuard Demo: JSON-RPC Function Call")
    print("=" * 60)

    # Define schema
    schema = {
        "type": "object",
        "properties": {
            "jsonrpc": {
                "type": "string",
                "const": "2.0"  # Must be exactly "2.0"
            },
            "method": {
                "type": "string",
                "minLength": 1
            },
            "params": {
                "type": "object"
            },
            "id": {
                "type": "integer"
            }
        },
        "required": ["jsonrpc", "method", "id"]
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
        "Generate a JSON-RPC call to method 'getUserProfile' with userId parameter",
        "Create a JSON-RPC request for 'updateSettings' method",
        "Make a JSON-RPC call to 'getData' with id 123",
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
            max_tokens=150,
            max_retries=3
        )

        print(f"\n{'='*30}")
        print(f"Result:")
        print(f"{'='*30}")
        print(f"Valid: {'✓' if result.is_valid else '✗'} {result.is_valid}")
        print(f"Retries: {result.retries}")
        print(f"Latency: {result.latency_ms:.0f}ms")

        print(f"\nOutput:")
        if result.is_valid:
            try:
                rpc_call = json.loads(result.output)

                print(f"JSON-RPC Call:")
                print(f"  Version: {rpc_call.get('jsonrpc', 'N/A')}")
                print(f"  Method: {rpc_call.get('method', 'N/A')}")
                print(f"  ID: {rpc_call.get('id', 'N/A')}")

                if "params" in rpc_call:
                    print(f"  Params: {rpc_call['params']}")

                print(f"\nFull JSON:")
                print(json.dumps(rpc_call, indent=2))

                # Validate const constraint
                if rpc_call.get("jsonrpc") == "2.0":
                    print(f"\n✓ Const constraint validated: jsonrpc == '2.0'")
                else:
                    print(f"\n✗ Const constraint violated: jsonrpc != '2.0'")

            except json.JSONDecodeError:
                print(result.output)
        else:
            print(result.output)

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

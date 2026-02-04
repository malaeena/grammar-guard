"""
Command-line interface module.

This module provides a rich terminal interface for GrammarGuard using Typer and Rich.

Commands:
    - generate: Generate JSON conforming to a schema
    - validate: Validate existing JSON against a schema
    - benchmark: Run benchmarks comparing constrained vs unconstrained generation

Features:
    - Progress indicators during generation
    - Syntax-highlighted JSON output
    - Colored error messages with context
    - Statistics table (latency, tokens, retries)

Example Usage:
    ```bash
    # Basic generation
    grammar-guard generate \\
        --schema schema.json \\
        --prompt "Generate a user profile" \\
        --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \\
        --backend transformers

    # With options
    grammar-guard generate \\
        --schema user.json \\
        --prompt "Create a user" \\
        --model models/mistral-7b.gguf \\
        --backend llamacpp \\
        --retries 3 \\
        --temperature 0.7 \\
        --device mps \\
        --output result.json

    # Benchmark mode
    grammar-guard benchmark \\
        --schema schema.json \\
        --prompts prompts.txt \\
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --iterations 10
    ```
"""

from .main import app

__all__ = ["app"]

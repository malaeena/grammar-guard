"""
GrammarGuard: Constrained Decoding for Reliable JSON Output from LLMs

GrammarGuard is a lightweight, runtime-agnostic constrained decoding layer that ensures
LLM-generated outputs conform to a target JSON schema without relying solely on prompting
or extensive post-validation.

Key Features:
    - Schema-driven token masking during generation
    - Support for JSON Schema and Pydantic models
    - Automatic retry with schema simplification on validation failures
    - Backend support for both HuggingFace transformers and llama.cpp
    - Optimized for Apple Silicon with MPS support

Quick Start:
    ```python
    from grammar_guard import GrammarConstrainedGenerator
    from pydantic import BaseModel

    class User(BaseModel):
        name: str
        age: int
        email: str

    generator = GrammarConstrainedGenerator(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        backend="transformers",
        device="mps"
    )

    result = generator.generate(
        prompt="Generate a user profile for John Doe",
        schema=User,
        max_retries=3
    )
    print(result.output)
    ```

Architecture:
    1. Schema Parser: Convert JSON Schema/Pydantic → internal constraints
    2. FSM Builder: Build character-level FSM, then token-level validity index
    3. Logits Processor: Mask invalid tokens during generation
    4. Validator: Post-generation validation with error reporting
    5. Retry Logic: Automatic retry with progressive schema simplification

For more information, see README.md and docs/architecture.md
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Main API exports - these are the primary user-facing classes
from grammar_guard.api import GrammarConstrainedGenerator  # noqa: F401

# Future exports (will be uncommented as implemented):
# from grammar_guard.builder import GrammarGuardBuilder  # noqa: F401
# from grammar_guard.validation.validator import ValidationResult  # noqa: F401

__all__ = [
    "GrammarConstrainedGenerator",
    # "GrammarGuardBuilder",
    # "ValidationResult",
]

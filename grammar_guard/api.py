"""
High-level Python API for GrammarGuard.

This module provides the main user-facing API for constrained JSON generation.
"""

from grammar_guard.generator import GrammarConstrainedGenerator, GenerationResult

# Re-export for convenience
__all__ = ["GrammarConstrainedGenerator", "GenerationResult"]

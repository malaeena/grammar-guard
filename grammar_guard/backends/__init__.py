"""
Model backend abstraction module.

This module provides a unified interface for different LLM backends, allowing
GrammarGuard to work with both HuggingFace transformers and llama.cpp models.

Components:
    - base: Abstract Backend protocol defining the interface
    - transformers_backend: HuggingFace transformers implementation
    - llamacpp_backend: llama.cpp implementation with Metal support
    - model_loader: Utilities for loading models with device optimization
    - device_utils: Device detection (MPS, CUDA, CPU)

Backend Protocol:
    All backends implement a common interface:
    - load_model(model_id, device): Load and initialize model
    - generate(prompt, logits_processor, **kwargs): Generate with constraints
    - get_tokenizer(): Get tokenizer for vocabulary analysis

Apple Silicon Optimization:
    Both backends are optimized for Apple Silicon:
    - transformers: Uses MPS (Metal Performance Shaders) via torch.device("mps")
    - llama.cpp: Uses Metal acceleration via n_gpu_layers=-1

Example:
    ```python
    from grammar_guard.backends import TransformersBackend

    # Load model on Apple Silicon
    backend = TransformersBackend(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device="mps"  # Metal Performance Shaders
    )

    # Generate with constrained decoding
    output = backend.generate(
        prompt="Generate a user profile",
        logits_processor=processor,
        max_tokens=100
    )
    ```
"""

from grammar_guard.backends.base import Backend, BackendFactory
from grammar_guard.backends.device_utils import (
    get_optimal_device,
    is_mps_available,
    is_cuda_available,
    is_apple_silicon,
    get_device_info,
    print_device_info
)
from grammar_guard.backends.transformers_backend import TransformersBackend
from grammar_guard.backends.llamacpp_backend import LlamaCppBackend

__all__ = [
    "Backend",
    "BackendFactory",
    "TransformersBackend",
    "LlamaCppBackend",
    "get_optimal_device",
    "is_mps_available",
    "is_cuda_available",
    "is_apple_silicon",
    "get_device_info",
    "print_device_info",
]

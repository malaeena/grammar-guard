"""
Backend abstraction - unified interface for different LLM runtimes.

This module defines the abstract Backend protocol that all model backends must
implement. This allows GrammarGuard to work with different LLM runtimes
(HuggingFace transformers, llama.cpp, etc.) through a common interface.

Backend Protocol:
    - load_model(): Initialize model with device optimization
    - generate(): Generate text with constrained decoding
    - get_tokenizer(): Get tokenizer for vocabulary analysis
    - get_model_info(): Get model metadata

Benefits:
    - Users can switch backends without changing code
    - Easy to add new backends (vLLM, GGML, etc.)
    - Consistent API across different model formats

Usage:
    ```python
    from grammar_guard.backends import TransformersBackend, LlamaCppBackend

    # Use HuggingFace transformers
    backend = TransformersBackend("gpt2", device="mps")

    # Or use llama.cpp
    backend = LlamaCppBackend("models/mistral-7b.gguf")

    # Same interface for both
    output = backend.generate(
        prompt="Generate JSON",
        logits_processor=processor,
        max_tokens=100
    )
    ```
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List


class Backend(ABC):
    """
    Abstract base class for LLM backends.

    All backends must implement these methods to be compatible with GrammarGuard.

    Attributes:
        model_id: Model identifier (HF model name or file path)
        device: Device to run on (cpu, cuda, mps)
        model: Loaded model instance
        tokenizer: Tokenizer instance
    """

    @abstractmethod
    def load_model(self, model_id: str, device: Optional[str] = None, **kwargs) -> None:
        """
        Load and initialize the model.

        Args:
            model_id: Model identifier (HF name or file path)
            device: Device to load on (cpu, cuda, mps, or None for auto-detect)
            **kwargs: Backend-specific options

        Raises:
            ValueError: If model cannot be loaded
            RuntimeError: If device is not available

        Example:
            ```python
            backend = TransformersBackend()
            backend.load_model(
                "gpt2",
                device="mps",
                torch_dtype=torch.float16
            )
            ```
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        logits_processor: Optional[Any] = None,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop_strings: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Generate text with optional constrained decoding.

        Args:
            prompt: Input prompt
            logits_processor: Optional LogitsProcessor for constrained decoding
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            stop_strings: List of strings that stop generation
            **kwargs: Backend-specific generation options

        Returns:
            str: Generated text

        Example:
            ```python
            output = backend.generate(
                prompt="Generate a user profile:",
                logits_processor=constrained_processor,
                max_tokens=100,
                temperature=0.7
            )
            ```
        """
        pass

    @abstractmethod
    def get_tokenizer(self) -> Any:
        """
        Get the tokenizer instance.

        Returns:
            Tokenizer instance (HF tokenizer or llama.cpp tokenizer)

        Example:
            ```python
            tokenizer = backend.get_tokenizer()
            tokens = tokenizer.encode("hello")
            ```
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata and information.

        Returns:
            Dict with model info:
                - model_id: Model identifier
                - device: Device model is on
                - vocab_size: Vocabulary size
                - context_length: Maximum context length
                - dtype: Model data type (if applicable)

        Example:
            ```python
            info = backend.get_model_info()
            print(f"Vocab size: {info['vocab_size']}")
            print(f"Device: {info['device']}")
            ```
        """
        pass

    def supports_constrained_decoding(self) -> bool:
        """
        Check if backend supports constrained decoding via logits processors.

        Returns:
            bool: True if constrained decoding is supported

        Note:
            Most backends support this, but some may have limitations.
        """
        return True

    def __repr__(self) -> str:
        info = self.get_model_info()
        return (
            f"{self.__class__.__name__}("
            f"model={info.get('model_id', 'unknown')}, "
            f"device={info.get('device', 'unknown')})"
        )


class BackendFactory:
    """
    Factory for creating backend instances.

    Automatically detects backend type from model_id and creates appropriate
    backend instance.

    Usage:
        ```python
        from grammar_guard.backends import BackendFactory

        # Auto-detect HuggingFace model
        backend = BackendFactory.create("gpt2", device="mps")

        # Auto-detect GGUF file
        backend = BackendFactory.create("models/mistral-7b.gguf")

        # Explicit backend type
        backend = BackendFactory.create("gpt2", backend_type="transformers")
        ```
    """

    @staticmethod
    def create(
        model_id: str,
        backend_type: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> Backend:
        """
        Create appropriate backend for model.

        Args:
            model_id: Model identifier or file path
            backend_type: Explicit backend type ("transformers", "llamacpp")
                         If None, auto-detect from model_id
            device: Device to use
            **kwargs: Backend-specific options

        Returns:
            Backend: Initialized backend instance

        Raises:
            ValueError: If backend type cannot be determined or is unsupported

        Example:
            ```python
            # Auto-detect
            backend = BackendFactory.create("gpt2")

            # Explicit
            backend = BackendFactory.create(
                "mistral-7b.gguf",
                backend_type="llamacpp",
                n_gpu_layers=-1
            )
            ```
        """
        if backend_type is None:
            backend_type = BackendFactory._detect_backend_type(model_id)

        if backend_type == "transformers":
            from grammar_guard.backends.transformers_backend import TransformersBackend
            backend = TransformersBackend(model_id, device=device, **kwargs)

        elif backend_type == "llamacpp":
            from grammar_guard.backends.llamacpp_backend import LlamaCppBackend
            backend = LlamaCppBackend(model_id, **kwargs)

        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")

        return backend

    @staticmethod
    def _detect_backend_type(model_id: str) -> str:
        """
        Auto-detect backend type from model identifier.

        Args:
            model_id: Model identifier or file path

        Returns:
            str: Backend type ("transformers" or "llamacpp")

        Strategy:
            - If ends with .gguf or .ggml → llamacpp
            - If contains '/' (HF format) → transformers
            - Otherwise → transformers (default)
        """
        model_id_lower = model_id.lower()

        # GGUF/GGML files → llama.cpp
        if model_id_lower.endswith(('.gguf', '.ggml', '.bin')):
            return "llamacpp"

        # HuggingFace format (org/model) → transformers
        if '/' in model_id and not model_id.startswith(('/','.')):
            return "transformers"

        # Default to transformers
        return "transformers"

    @staticmethod
    def list_available_backends() -> List[str]:
        """
        List available backends on this system.

        Returns:
            List of backend names that can be used

        Example:
            ```python
            backends = BackendFactory.list_available_backends()
            print(f"Available: {', '.join(backends)}")
            ```
        """
        available = []

        # Check transformers
        try:
            import transformers
            available.append("transformers")
        except ImportError:
            pass

        # Check llama-cpp-python
        try:
            import llama_cpp
            available.append("llamacpp")
        except ImportError:
            pass

        return available

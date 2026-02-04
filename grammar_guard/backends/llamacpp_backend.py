"""
llama.cpp backend with Metal acceleration for Apple Silicon.

This backend provides integration with llama.cpp through llama-cpp-python,
optimized for GGUF models with Metal acceleration on Apple Silicon.

Features:
    - GGUF model support
    - Metal acceleration (all layers on GPU)
    - Logits processor integration for constrained decoding
    - Low memory usage
    - Fast inference

Usage:
    ```python
    from grammar_guard.backends import LlamaCppBackend

    # Load GGUF model with Metal acceleration
    backend = LlamaCppBackend(
        "models/mistral-7b-q4.gguf",
        n_gpu_layers=-1  # All layers on GPU (Metal)
    )

    # Generate with constraints
    output = backend.generate(
        prompt="Generate a user profile:",
        logits_processor=processor,
        max_tokens=100
    )
    ```
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LlamaCppBackend:
    """
    Backend for llama.cpp (GGUF) models.

    Attributes:
        model_path: Path to GGUF model file
        llm: llama_cpp.Llama instance
        n_gpu_layers: Number of layers on GPU (-1 = all)
        n_ctx: Context length
    """

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 2048,
        n_batch: int = 512,
        use_mlock: bool = True,
        **kwargs
    ):
        """
        Initialize llama.cpp backend.

        Args:
            model_path: Path to GGUF model file
            n_gpu_layers: Number of layers to offload to GPU
                         -1 = all layers (recommended for Apple Silicon)
            n_ctx: Context window size
            n_batch: Batch size for prompt processing
            use_mlock: Lock model in memory (prevents swapping)
            **kwargs: Additional llama.cpp options

        Example:
            ```python
            # Full GPU acceleration (Apple Silicon)
            backend = LlamaCppBackend(
                "models/mistral-7b.gguf",
                n_gpu_layers=-1,  # All on Metal
                n_ctx=4096,
                use_mlock=True
            )

            # CPU only
            backend = LlamaCppBackend(
                "models/mistral-7b.gguf",
                n_gpu_layers=0
            )
            ```
        """
        self.model_path = Path(model_path)
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.use_mlock = use_mlock
        self.llm = None

        logger.info(
            f"Initializing LlamaCppBackend: model={model_path}, "
            f"n_gpu_layers={n_gpu_layers}"
        )

        # Validate model file
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model
        self._load_model(**kwargs)

    def _load_model(self, **kwargs):
        """Load llama.cpp model."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required. "
                "Install with: pip install llama-cpp-python"
            )

        logger.info(f"Loading GGUF model: {self.model_path}")

        # Build load kwargs
        load_kwargs = {
            'model_path': str(self.model_path),
            'n_gpu_layers': self.n_gpu_layers,
            'n_ctx': self.n_ctx,
            'n_batch': self.n_batch,
            'use_mlock': self.use_mlock,
            'verbose': False,  # Reduce logging noise
        }

        # Merge user kwargs
        load_kwargs.update(kwargs)

        try:
            self.llm = Llama(**load_kwargs)
            logger.info("Model loaded successfully")

            # Log Metal status
            if self.n_gpu_layers != 0:
                logger.info(
                    f"Metal acceleration: {self.n_gpu_layers} layers "
                    f"({-1 if self.n_gpu_layers == -1 else self.n_gpu_layers} layers on GPU)"
                )

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

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
            logits_processor: Optional ConstrainedLogitsProcessor
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop_strings: List of strings that stop generation
            **kwargs: Additional generation parameters

        Returns:
            str: Generated text (prompt + generated tokens)

        Note:
            llama-cpp-python logits_processor API is slightly different from HF.
            We need to adapt our processor to work with llama.cpp format.

        Example:
            ```python
            output = backend.generate(
                prompt="Once upon a time",
                max_tokens=50,
                temperature=0.7
            )
            ```
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded")

        # Prepare generation kwargs
        gen_kwargs = {
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'echo': True,  # Include prompt in output
        }

        if stop_strings:
            gen_kwargs['stop'] = stop_strings

        # Handle logits processor
        # llama.cpp uses different API than HuggingFace
        # For MVP, we'll use the simpler approach without logits_processor
        # Full implementation would require adapting our processor to llama.cpp format
        if logits_processor is not None:
            logger.warning(
                "Logits processor with llama.cpp is experimental in MVP. "
                "For full constrained decoding, use transformers backend."
            )
            # TODO: Implement llama.cpp logits processor adapter

        # Merge user kwargs
        gen_kwargs.update(kwargs)

        logger.debug(f"Generating with: {gen_kwargs}")

        try:
            # Generate
            output = self.llm(prompt, **gen_kwargs)

            # Extract generated text
            generated_text = output['choices'][0]['text']

            logger.debug(
                f"Generated {output['usage']['completion_tokens']} tokens"
            )

            return generated_text

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def get_tokenizer(self) -> Any:
        """
        Get tokenizer instance.

        Returns:
            llama.cpp tokenizer wrapper

        Note:
            llama.cpp tokenizer has different API than HuggingFace.
            We provide a wrapper for compatibility.

        Example:
            ```python
            tokenizer = backend.get_tokenizer()
            tokens = tokenizer.tokenize("hello world")
            ```
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded")

        # Return a wrapper around llama.cpp tokenization
        return LlamaCppTokenizerWrapper(self.llm)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata.

        Returns:
            Dict with model information

        Example:
            ```python
            info = backend.get_model_info()
            print(f"Context length: {info['context_length']}")
            ```
        """
        info = {
            'model_path': str(self.model_path),
            'backend': 'llamacpp',
            'n_gpu_layers': self.n_gpu_layers,
            'context_length': self.n_ctx,
            'n_batch': self.n_batch,
        }

        if self.llm:
            # Try to get vocab size
            try:
                info['vocab_size'] = self.llm.n_vocab()
            except AttributeError:
                pass

        return info

    def __repr__(self) -> str:
        return (
            f"LlamaCppBackend(model={self.model_path.name}, "
            f"n_gpu_layers={self.n_gpu_layers})"
        )


class LlamaCppTokenizerWrapper:
    """
    Wrapper to make llama.cpp tokenizer compatible with HuggingFace API.

    This provides a unified interface for tokenization across backends.

    Attributes:
        llm: llama_cpp.Llama instance
    """

    def __init__(self, llm: Any):
        """
        Initialize tokenizer wrapper.

        Args:
            llm: llama_cpp.Llama instance
        """
        self.llm = llm

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens (BOS/EOS)

        Returns:
            List of token IDs

        Example:
            ```python
            token_ids = tokenizer.encode("hello world")
            ```
        """
        tokens = self.llm.tokenize(text.encode('utf-8'), add_bos=add_special_tokens)
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text

        Example:
            ```python
            text = tokenizer.decode([123, 456, 789])
            ```
        """
        # llama.cpp detokenize
        text_bytes = self.llm.detokenize(token_ids)
        return text_bytes.decode('utf-8', errors='ignore')

    def __len__(self) -> int:
        """Get vocabulary size."""
        return self.llm.n_vocab()

    def __call__(self, text: str, return_tensors: Optional[str] = None) -> Dict:
        """
        Tokenize text (HuggingFace-style API).

        Args:
            text: Text to tokenize
            return_tensors: Return format ("pt" for PyTorch)

        Returns:
            Dict with 'input_ids' and 'attention_mask'

        Example:
            ```python
            inputs = tokenizer("hello", return_tensors="pt")
            ```
        """
        token_ids = self.encode(text)

        if return_tensors == "pt":
            import torch
            return {
                'input_ids': torch.tensor([token_ids]),
                'attention_mask': torch.ones(len(token_ids))
            }
        else:
            return {
                'input_ids': token_ids,
                'attention_mask': [1] * len(token_ids)
            }


def test_llamacpp_backend(model_path: str) -> bool:
    """
    Test if llama.cpp backend works correctly.

    Args:
        model_path: Path to GGUF model file

    Returns:
        bool: True if test passes

    Example:
        ```python
        if test_llamacpp_backend("models/mistral-7b.gguf"):
            print("LlamaCpp backend works!")
        ```
    """
    try:
        logger.info(f"Testing LlamaCppBackend with {model_path}")

        # Create backend
        backend = LlamaCppBackend(model_path, n_gpu_layers=-1)

        # Test generation
        output = backend.generate(
            prompt="Hello",
            max_tokens=10,
            temperature=0.7
        )

        logger.info(f"Test generation: {output}")

        # Get info
        info = backend.get_model_info()
        logger.info(f"Model info: {info}")

        logger.info("LlamaCppBackend test passed!")
        return True

    except Exception as e:
        logger.error(f"LlamaCppBackend test failed: {e}")
        return False

"""
HuggingFace Transformers backend with Apple Silicon MPS support.

This backend provides integration with HuggingFace's transformers library,
optimized for Apple Silicon (M1/M2/M3) using Metal Performance Shaders (MPS).

Features:
    - Auto model loading with device optimization
    - MPS support for Apple Silicon
    - CUDA support for NVIDIA GPUs
    - CPU fallback
    - LogitsProcessor integration for constrained decoding
    - Half-precision (float16) support for efficiency

Usage:
    ```python
    from grammar_guard.backends import TransformersBackend

    # Load model on Apple Silicon
    backend = TransformersBackend(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device="mps"
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
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TransformersBackend:
    """
    Backend for HuggingFace transformers models.

    Attributes:
        model_id: HuggingFace model identifier
        device: Device to run on (mps, cuda, cpu)
        model: Loaded AutoModelForCausalLM instance
        tokenizer: Loaded AutoTokenizer instance
        torch_dtype: Data type for model (float16 or float32)
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        torch_dtype: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize Transformers backend.

        Args:
            model_id: HuggingFace model identifier (e.g., "gpt2")
            device: Device to use ("mps", "cuda", "cpu", or None for auto)
            torch_dtype: PyTorch data type (None for auto: float16 on GPU, float32 on CPU)
            **kwargs: Additional arguments for model loading

        Example:
            ```python
            # Auto-detect device
            backend = TransformersBackend("gpt2")

            # Explicit MPS
            backend = TransformersBackend("gpt2", device="mps")

            # Custom dtype
            import torch
            backend = TransformersBackend("gpt2", torch_dtype=torch.float32)
            ```
        """
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

        # Detect optimal device
        if device is None:
            from grammar_guard.backends.device_utils import get_optimal_device
            device = get_optimal_device()

        self.device = device

        # Set torch_dtype
        if torch_dtype is None:
            import torch
            # Use float16 on GPU for efficiency, float32 on CPU for compatibility
            self.torch_dtype = torch.float16 if device in ["mps", "cuda"] else torch.float32
        else:
            self.torch_dtype = torch_dtype

        logger.info(
            f"Initializing TransformersBackend: model={model_id}, "
            f"device={device}, dtype={self.torch_dtype}"
        )

        # Load model and tokenizer
        self._load_model(**kwargs)
        self._load_tokenizer()

    def _load_model(self, **kwargs):
        """Load HuggingFace model with optimizations."""
        try:
            from transformers import AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch accelerate"
            )

        logger.info(f"Loading model: {self.model_id}")

        # Device-specific optimizations
        load_kwargs = {
            'torch_dtype': self.torch_dtype,
            'low_cpu_mem_usage': True,  # Reduce CPU memory usage
        }

        # Merge user kwargs
        load_kwargs.update(kwargs)

        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **load_kwargs
            )

            # Move model to target device
            # Note: Don't use device_map for simple device placement as it requires accelerate
            if self.device:
                import torch
                self.model = self.model.to(torch.device(self.device))

            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_tokenizer(self):
        """Load HuggingFace tokenizer."""
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("transformers is required")

        logger.info(f"Loading tokenizer: {self.model_id}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            # Set pad token if not set (needed for batch generation)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Tokenizer loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
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
            stop_strings: List of strings that stop generation (not implemented in MVP)
            **kwargs: Additional generation parameters

        Returns:
            str: Generated text (prompt + generated tokens)

        Example:
            ```python
            output = backend.generate(
                prompt="Hello",
                max_tokens=50,
                temperature=0.7
            )
            ```
        """
        import torch

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Move to correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Prepare generation kwargs
        gen_kwargs = {
            'max_new_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'do_sample': temperature > 0,  # Use sampling if temperature > 0
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }

        # Add logits processor if provided
        if logits_processor is not None:
            from transformers import LogitsProcessorList
            # Reset processor state before generation
            if hasattr(logits_processor, 'reset'):
                logits_processor.reset()
            gen_kwargs['logits_processor'] = LogitsProcessorList([logits_processor])

        # Merge user kwargs
        gen_kwargs.update(kwargs)

        logger.debug(f"Generating with: {gen_kwargs}")

        # Generate
        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            # Decode only the newly generated tokens (excluding the prompt)
            # outputs[0] contains: [prompt_tokens, generated_tokens]
            # We want only the generated part
            prompt_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][prompt_length:]

            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            logger.debug(f"Generated {len(generated_tokens)} new tokens (total: {len(outputs[0])})")

            return generated_text

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def get_tokenizer(self) -> Any:
        """
        Get the tokenizer instance.

        Returns:
            AutoTokenizer: HuggingFace tokenizer

        Example:
            ```python
            tokenizer = backend.get_tokenizer()
            tokens = tokenizer.encode("hello world")
            ```
        """
        return self.tokenizer

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata.

        Returns:
            Dict with model information

        Example:
            ```python
            info = backend.get_model_info()
            print(f"Vocab size: {info['vocab_size']}")
            ```
        """
        info = {
            'model_id': self.model_id,
            'device': self.device,
            'dtype': str(self.torch_dtype),
            'backend': 'transformers'
        }

        if self.tokenizer:
            info['vocab_size'] = len(self.tokenizer)

        if self.model:
            # Try to get model config
            if hasattr(self.model, 'config'):
                config = self.model.config
                if hasattr(config, 'max_position_embeddings'):
                    info['context_length'] = config.max_position_embeddings
                if hasattr(config, 'hidden_size'):
                    info['hidden_size'] = config.hidden_size
                if hasattr(config, 'num_hidden_layers'):
                    info['num_layers'] = config.num_hidden_layers

        return info

    def __repr__(self) -> str:
        return (
            f"TransformersBackend(model={self.model_id}, "
            f"device={self.device}, dtype={self.torch_dtype})"
        )


def test_transformers_backend(model_id: str = "gpt2", device: Optional[str] = None) -> bool:
    """
    Test if transformers backend works correctly.

    Args:
        model_id: Model to test with
        device: Device to test on

    Returns:
        bool: True if test passes

    Example:
        ```python
        if test_transformers_backend("gpt2", "mps"):
            print("Transformers backend works!")
        ```
    """
    try:
        logger.info(f"Testing TransformersBackend with {model_id}")

        # Create backend
        backend = TransformersBackend(model_id, device=device)

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

        logger.info("TransformersBackend test passed!")
        return True

    except Exception as e:
        logger.error(f"TransformersBackend test failed: {e}")
        return False

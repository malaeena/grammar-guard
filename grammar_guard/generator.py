"""
Main constrained generation orchestrator with retry logic.

This is the core class that ties all components together:
    1. Parse schema → regex → FSM → token index
    2. Load model backend
    3. Create constrained logits processor
    4. Generate with constraints
    5. Validate output
    6. On failure: simplify schema and retry

Usage:
    ```python
    from grammar_guard import GrammarConstrainedGenerator

    generator = GrammarConstrainedGenerator(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        backend="transformers",
        device="mps"
    )

    result = generator.generate(
        prompt="Generate a user profile",
        schema={"type": "object", "properties": {"name": {"type": "string"}}},
        max_retries=3
    )

    print(result.output)
    ```
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """
    Result of a generation attempt.

    Attributes:
        output: Generated text
        is_valid: Whether output is valid
        retries: Number of retries used
        simplification_level: Schema simplification level used
        latency_ms: Generation time in milliseconds
        tokens_generated: Number of tokens generated
        validation_errors: List of validation errors (if invalid)
    """
    output: str
    is_valid: bool
    retries: int
    simplification_level: int
    latency_ms: float
    tokens_generated: int
    validation_errors: List[Any]


class GrammarConstrainedGenerator:
    """
    Main class for constrained JSON generation.

    This orchestrates all components to generate JSON that conforms to a schema.

    Attributes:
        model_id: Model identifier
        backend_type: Backend type ("transformers" or "llamacpp")
        device: Device to use
        backend: Loaded backend instance
    """

    def __init__(
        self,
        model: str,
        backend: str = "transformers",
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize constrained generator.

        Args:
            model: Model identifier or path
            backend: Backend type ("transformers" or "llamacpp")
            device: Device to use (None for auto-detect)
            **kwargs: Additional backend options

        Example:
            ```python
            # Transformers on Apple Silicon
            gen = GrammarConstrainedGenerator(
                "gpt2",
                backend="transformers",
                device="mps"
            )

            # llama.cpp with GGUF
            gen = GrammarConstrainedGenerator(
                "models/mistral-7b.gguf",
                backend="llamacpp"
            )
            ```
        """
        self.model_id = model
        self.backend_type = backend
        self.device = device

        logger.info(
            f"Initializing GrammarConstrainedGenerator: "
            f"model={model}, backend={backend}, device={device}"
        )

        # Load backend
        from grammar_guard.backends import BackendFactory
        self.backend = BackendFactory.create(
            model_id=model,
            backend_type=backend,
            device=device,
            **kwargs
        )

        logger.info("Backend loaded successfully")

    def generate(
        self,
        prompt: str,
        schema: Union[Dict, type],
        max_tokens: int = 200,
        max_retries: int = 3,
        temperature: float = 0.7,
        **kwargs
    ) -> GenerationResult:
        """
        Generate JSON conforming to schema with retry on failure.

        Args:
            prompt: Input prompt
            schema: JSON Schema dict or Pydantic model
            max_tokens: Maximum tokens to generate
            max_retries: Maximum retry attempts
            temperature: Sampling temperature
            **kwargs: Additional generation options

        Returns:
            GenerationResult: Result with output and metadata

        Strategy:
            1. Try with original schema
            2. On validation failure: simplify schema
            3. Retry up to max_retries times
            4. Track simplifications applied

        Example:
            ```python
            result = generator.generate(
                prompt="Create a user",
                schema={"type": "object", "properties": {"name": {"type": "string"}}},
                max_retries=3
            )

            if result.is_valid:
                print(f"Success! Output: {result.output}")
            else:
                print(f"Failed after {result.retries} retries")
            ```
        """
        start_time = time.time()

        # Parse schema
        from grammar_guard.schema import parse_schema
        from grammar_guard.schema.simplifier import simplify_schema

        original_schema = schema if isinstance(schema, dict) else None
        if original_schema is None:
            # Pydantic model
            from grammar_guard.schema import pydantic_to_schema
            original_schema = pydantic_to_schema(schema)

        current_schema = original_schema
        simplification_level = 0

        # Try generation with progressive simplification
        for attempt in range(max_retries + 1):
            logger.info(
                f"Generation attempt {attempt + 1}/{max_retries + 1} "
                f"(simplification level: {simplification_level})"
            )

            try:
                # Generate with current schema
                output = self._generate_with_schema(
                    prompt=prompt,
                    schema=current_schema,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )

                # Validate
                from grammar_guard.validation import validate
                validation_result = validate(output, current_schema)

                # Calculate stats
                latency_ms = (time.time() - start_time) * 1000
                tokenizer = self.backend.get_tokenizer()
                tokens = tokenizer.encode(output, add_special_tokens=False)
                tokens_generated = len(tokens)

                if validation_result.is_valid:
                    # Success!
                    logger.info(
                        f"Generation successful after {attempt} retries "
                        f"(simplification level: {simplification_level})"
                    )

                    return GenerationResult(
                        output=output,
                        is_valid=True,
                        retries=attempt,
                        simplification_level=simplification_level,
                        latency_ms=latency_ms,
                        tokens_generated=tokens_generated,
                        validation_errors=[]
                    )

                else:
                    # Validation failed
                    logger.warning(
                        f"Validation failed with {len(validation_result.errors)} errors"
                    )

                    if attempt < max_retries:
                        # Simplify schema for next attempt
                        simplification_level += 1
                        simplified_schema, changes = simplify_schema(
                            current_schema,
                            level=simplification_level
                        )

                        logger.info(
                            f"Simplifying schema to level {simplification_level}: "
                            f"{len(changes)} changes"
                        )
                        for change in changes:
                            logger.debug(f"  - {change}")

                        current_schema = simplified_schema

                    else:
                        # Max retries reached
                        logger.error("Max retries reached, generation failed")

                        return GenerationResult(
                            output=output,
                            is_valid=False,
                            retries=attempt,
                            simplification_level=simplification_level,
                            latency_ms=latency_ms,
                            tokens_generated=tokens_generated,
                            validation_errors=validation_result.errors
                        )

            except Exception as e:
                logger.error(f"Generation attempt {attempt + 1} failed: {e}")

                if attempt == max_retries:
                    # Failed completely
                    return GenerationResult(
                        output="",
                        is_valid=False,
                        retries=attempt,
                        simplification_level=simplification_level,
                        latency_ms=(time.time() - start_time) * 1000,
                        tokens_generated=0,
                        validation_errors=[{"error": str(e)}]
                    )

                # Try again with simplified schema
                simplification_level += 1
                simplified_schema, _ = simplify_schema(
                    current_schema,
                    level=simplification_level
                )
                current_schema = simplified_schema

        # Should never reach here
        return GenerationResult(
            output="",
            is_valid=False,
            retries=max_retries,
            simplification_level=simplification_level,
            latency_ms=(time.time() - start_time) * 1000,
            tokens_generated=0,
            validation_errors=[{"error": "Unknown error"}]
        )

    def _generate_with_schema(
        self,
        prompt: str,
        schema: Dict,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> str:
        """
        Internal method to generate with a specific schema.

        This builds the FSM, token index, and logits processor for the schema,
        then generates with constraints.

        Args:
            prompt: Input prompt
            schema: JSON Schema dict
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional options

        Returns:
            str: Generated output
        """
        logger.debug("Building constrained decoding components...")

        # Parse schema to constraint
        from grammar_guard.schema import parse_schema, compile_to_regex

        constraint = parse_schema(schema)

        # Compile to regex
        regex = compile_to_regex(constraint)
        logger.debug(f"Regex pattern: {regex[:100]}...")

        # Build FSM
        from interegular import parse_pattern

        try:
            char_fsm = parse_pattern(regex).to_fsm()
            logger.debug(f"FSM built: {len(char_fsm.states)} states")
        except Exception as e:
            logger.error(f"Failed to build FSM: {e}")
            raise

        # Build token index (with caching)
        from grammar_guard.decoding import build_token_index, TokenIndex
        from grammar_guard.decoding.cache import get_cache_manager, compute_cache_key

        cache_manager = get_cache_manager()
        tokenizer = self.backend.get_tokenizer()

        # Try to load from cache
        cache_key = compute_cache_key(schema, self.model_id)
        cached_index = cache_manager.load_index(cache_key)

        if cached_index is not None:
            logger.info("Using cached token index")
            token_index = cached_index
        else:
            logger.info("Building token index (this may take a minute)...")
            raw_index = build_token_index(char_fsm, tokenizer)
            token_index = TokenIndex(raw_index, vocab_size=len(tokenizer))
            token_index = token_index.optimize()

            # Cache for future use
            cache_manager.save_index(cache_key, token_index)
            logger.info("Token index cached")

        # Get prompt length for skipping prompt tokens during validation
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_length = len(prompt_tokens)

        # Create logits processor
        from grammar_guard.decoding import ConstrainedLogitsProcessor

        logits_processor = ConstrainedLogitsProcessor(
            token_index=token_index,
            fsm=char_fsm,
            tokenizer=tokenizer,
            prompt_length=prompt_length
        )

        logger.debug("Constrained decoding components ready")

        # Generate with backend
        output = self.backend.generate(
            prompt=prompt,
            logits_processor=logits_processor,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        return output

    def get_info(self) -> Dict[str, Any]:
        """Get generator information."""
        info = {
            'model': self.model_id,
            'backend': self.backend_type,
            'device': self.device,
        }

        if self.backend:
            info.update(self.backend.get_model_info())

        return info

    def __repr__(self) -> str:
        return (
            f"GrammarConstrainedGenerator(model={self.model_id}, "
            f"backend={self.backend_type}, device={self.device})"
        )

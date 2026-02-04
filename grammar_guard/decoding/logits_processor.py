"""
Logits Processor - mask invalid tokens during LLM generation.

This module implements the HuggingFace LogitsProcessor protocol to apply
constrained decoding. During generation, the model produces logits (scores)
for each token in the vocabulary. We mask invalid tokens by setting their
logits to -inf, forcing the model to only choose valid tokens.

Flow:
    1. Model generates logits for all tokens
    2. LogitsProcessor is called with (input_ids, logits)
    3. We determine current FSM state from input_ids
    4. We look up valid tokens for that state in token_index
    5. We set logits for invalid tokens to -inf
    6. Model samples from masked logits (only valid tokens possible)

This is the critical integration point between our FSM constraints and
the LLM generation process.

Usage:
    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from grammar_guard.decoding import ConstrainedLogitsProcessor

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Create processor
    processor = ConstrainedLogitsProcessor(token_index, fsm, tokenizer)

    # Generate with constraints
    output = model.generate(
        input_ids,
        logits_processor=[processor],
        max_new_tokens=100
    )
    ```
"""

import logging
from typing import Any, Optional, Set

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class ConstrainedLogitsProcessor:
    """
    LogitsProcessor that enforces JSON schema constraints via token masking.

    This implements the HuggingFace LogitsProcessor interface:
        __call__(input_ids: Tensor, scores: Tensor) -> Tensor

    The processor:
    1. Tracks current FSM state based on generated tokens
    2. Looks up valid tokens for current state
    3. Masks invalid tokens (sets logits to -inf)
    4. Handles special tokens (EOS, PAD, etc.)

    Attributes:
        token_index: Token validity index {(state, token): is_valid}
        fsm: Character-level FSM for state tracking
        tokenizer: Tokenizer for decoding tokens
        state_tracker: Tracks current FSM state during generation
        allow_eos_when_valid: Allow EOS token when in accepting state
    """

    def __init__(
        self,
        token_index: Any,  # TokenIndex
        fsm: Any,  # interegular FSM
        tokenizer: Any,
        allow_eos_when_valid: bool = True,
        special_tokens: Optional[Set[int]] = None,
        prompt_length: int = 0
    ):
        """
        Initialize ConstrainedLogitsProcessor.

        Args:
            token_index: TokenIndex with (state, token) → valid mapping
            fsm: Character-level FSM for state tracking
            tokenizer: Tokenizer for decoding token IDs
            allow_eos_when_valid: Allow EOS token when in accepting FSM state
            special_tokens: Set of special token IDs to handle specially
            prompt_length: Number of prompt tokens to skip during validation
        """
        self.token_index = token_index
        self.fsm = fsm
        self.tokenizer = tokenizer
        self.allow_eos_when_valid = allow_eos_when_valid

        from grammar_guard.decoding.state_tracker import FSMStateTracker
        self.state_tracker = FSMStateTracker(fsm, tokenizer)

        if special_tokens is None:
            special_tokens = self._get_special_tokens()
        self.special_tokens = special_tokens

        self.eos_token_id = getattr(tokenizer, 'eos_token_id', None)
        self.pad_token_id = getattr(tokenizer, 'pad_token_id', None)

        self.persistent_state = self.state_tracker.get_initial_state()
        self.prompt_length = prompt_length
        self.last_processed_length = prompt_length

        logger.debug(
            f"ConstrainedLogitsProcessor initialized "
            f"(vocab={token_index.vocab_size}, EOS={self.eos_token_id}, prompt_length={prompt_length})"
        )

    def _get_special_tokens(self) -> Set[int]:
        """Get set of special token IDs from tokenizer."""
        special = set()

        # HuggingFace tokenizer
        if hasattr(self.tokenizer, 'all_special_ids'):
            special.update(self.tokenizer.all_special_ids)

        # Individual special tokens
        for attr in ['eos_token_id', 'bos_token_id', 'pad_token_id', 'unk_token_id']:
            token_id = getattr(self.tokenizer, attr, None)
            if token_id is not None:
                special.add(token_id)

        return special

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        """
        Apply constrained decoding by masking invalid tokens.

        This is called by HuggingFace generate() after model computes logits.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len) with generated tokens
            scores: Tensor of shape (batch_size, vocab_size) with logits

        Returns:
            Tensor: Modified scores with invalid tokens masked to -inf

        Note:
            - We modify scores in-place for efficiency
            - Setting logit to -inf makes probability ≈ 0 after softmax
            - This forces model to only sample from valid tokens

        Example:
            ```python
            # During generation, HuggingFace calls:
            input_ids = torch.tensor([[1, 2, 3, 4]])  # Generated so far
            scores = model(input_ids).logits[:, -1, :]  # Next token logits

            # Processor masks invalid tokens
            masked_scores = processor(input_ids, scores)

            # Model samples from masked scores
            next_token = torch.argmax(masked_scores)  # or sample
            ```
        """
        batch_size = input_ids.shape[0]

        # Process each sequence in batch
        for batch_idx in range(batch_size):
            # Get current sequence
            seq = input_ids[batch_idx]

            # Determine current FSM state
            current_state = self._get_current_state(seq)

            # Get valid tokens for this state
            valid_tokens = self._get_valid_tokens(current_state)

            # Mask invalid tokens
            mask = self._create_mask(valid_tokens, scores.shape[1], scores.device)
            scores[batch_idx, mask] = float('-inf')

            # Handle EOS token
            if self.eos_token_id is not None:
                if self.allow_eos_when_valid and self.state_tracker.is_accepting(current_state):
                    # Allow EOS when in accepting state
                    scores[batch_idx, self.eos_token_id] = scores[batch_idx, self.eos_token_id]
                else:
                    # Block EOS when not in accepting state
                    scores[batch_idx, self.eos_token_id] = float('-inf')

        return scores

    def _get_current_state(self, input_ids: Tensor) -> int:
        """
        Determine current FSM state from generated tokens using incremental tracking.

        Args:
            input_ids: Tensor of token IDs (prompt + generated)

        Returns:
            int: Current FSM state

        Strategy:
            Maintains persistent state between calls. Only processes NEW tokens
            since the last call, avoiding O(n²) replay overhead. Skips prompt
            tokens by position.

        Example:
            Prompt length: 7 tokens
            Call 1: input_ids=[1,2,3,4,5,6,7] (prompt only) → state=0
            Call 2: input_ids=[1,2,3,4,5,6,7,15] → process [15], state=3
            Call 3: input_ids=[1,2,3,4,5,6,7,15,20] → process [20], state=8
        """
        current_length = len(input_ids)

        if current_length <= self.last_processed_length:
            return self.persistent_state

        start_idx = max(self.last_processed_length, self.prompt_length)
        new_tokens = input_ids[start_idx:current_length].tolist()

        for token_id in new_tokens:
            if token_id in self.special_tokens:
                continue

            try:
                self.persistent_state = self.state_tracker.update(
                    self.persistent_state, token_id
                )
            except ValueError:
                pass

        self.last_processed_length = current_length
        return self.persistent_state

    def _get_valid_tokens(self, state: int) -> Set[int]:
        """
        Get set of valid token IDs for given FSM state.

        Args:
            state: FSM state ID

        Returns:
            Set of valid token IDs

        Example:
            ```python
            valid_tokens = self._get_valid_tokens(state=5)
            # Returns: {10, 15, 234, ...}  (all valid tokens in state 5)
            ```
        """
        return self.token_index.get_valid_tokens(state)

    def _create_mask(
        self,
        valid_tokens: Set[int],
        vocab_size: int,
        device: torch.device
    ) -> Tensor:
        """
        Create boolean mask for invalid tokens.

        Args:
            valid_tokens: Set of valid token IDs
            vocab_size: Size of vocabulary
            device: Torch device (cpu/cuda/mps)

        Returns:
            Tensor: Boolean mask where True = invalid (should be masked)

        Example:
            ```python
            valid_tokens = {1, 5, 10}
            mask = self._create_mask(valid_tokens, vocab_size=100, device='cpu')
            # mask[1] = False (valid)
            # mask[2] = True (invalid)
            # mask[5] = False (valid)
            # ... etc
            ```
        """
        # Create mask of all True (all invalid initially)
        mask = torch.ones(vocab_size, dtype=torch.bool, device=device)

        # Set valid tokens to False (don't mask)
        if valid_tokens:
            valid_indices = torch.tensor(list(valid_tokens), device=device, dtype=torch.long)
            mask[valid_indices] = False

        return mask

    def reset(self) -> None:
        """
        Reset processor state for new generation.

        Call this between generation calls if reusing the processor.

        Example:
            ```python
            # Generate first sequence
            output1 = model.generate(..., logits_processor=[processor])

            # Reset for next sequence
            processor.reset()

            # Generate second sequence
            output2 = model.generate(..., logits_processor=[processor])
            ```
        """
        self.state_tracker.reset()
        logger.debug("ConstrainedLogitsProcessor reset")

    def get_stats(self) -> dict:
        """
        Get statistics about processor usage.

        Returns:
            Dict with statistics

        Example:
            ```python
            stats = processor.get_stats()
            print(f"Current state: {stats['current_state']}")
            print(f"Is accepting: {stats['is_accepting']}")
            ```
        """
        return {
            'current_state': self.state_tracker.current_state,
            'is_accepting': self.state_tracker.is_accepting(self.state_tracker.current_state),
            'vocab_size': self.token_index.vocab_size,
            'num_states': self.token_index.num_states
        }

    def __repr__(self) -> str:
        return (
            f"ConstrainedLogitsProcessor("
            f"state={self.state_tracker.current_state}, "
            f"vocab={self.token_index.vocab_size})"
        )


class ConstrainedLogitsProcessorList:
    """
    Convenience wrapper for using multiple logits processors.

    HuggingFace generate() accepts a list of processors. This class
    wraps our constrained processor with any other processors you want.

    Example:
        ```python
        from transformers import TemperatureLogitsWarper

        processors = ConstrainedLogitsProcessorList(
            constrained_processor,
            TemperatureLogitsWarper(temperature=0.8)
        )

        output = model.generate(..., logits_processor=processors)
        ```
    """

    def __init__(self, *processors):
        """
        Initialize with list of processors.

        Args:
            *processors: Variable number of LogitsProcessor instances
        """
        self.processors = list(processors)

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        """Apply all processors in sequence."""
        for processor in self.processors:
            scores = processor(input_ids, scores)
        return scores

    def append(self, processor) -> None:
        """Add a processor to the list."""
        self.processors.append(processor)

    def __len__(self) -> int:
        return len(self.processors)

    def __repr__(self) -> str:
        return f"ConstrainedLogitsProcessorList({len(self.processors)} processors)"

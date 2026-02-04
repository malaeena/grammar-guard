"""
Constrained decoding engine module.

This module implements the core constrained decoding logic that masks invalid tokens
during LLM generation to enforce JSON schema constraints.

Components:
    - fsm_builder: Convert character-level FSM to token-level validity index
    - token_index: Token validity index for O(1) lookup during generation
    - logits_processor: HuggingFace LogitsProcessor implementation for token masking
    - state_tracker: Track FSM state as tokens are generated
    - cache: Cache compiled schemas and token indices for performance

Key Algorithm - Token Index Building:
    For each (FSM state, token) pair:
    1. Decode token to string
    2. Try to transition through FSM with that string
    3. Mark token as valid if transition succeeds
    4. Build index: {(state_id, token_id): is_valid}

This pre-computation allows O(1) lookup during generation, making constrained
decoding fast enough for real-time use.

Example:
    ```python
    from grammar_guard.decoding import build_token_index, ConstrainedLogitsProcessor
    from interegular import parse_pattern

    # Build character-level FSM from regex
    regex = r'\{"name":"[^"]*"\}'
    char_fsm = parse_pattern(regex).to_fsm()

    # Build token-level index
    token_index = build_token_index(char_fsm, tokenizer)

    # Use in generation
    processor = ConstrainedLogitsProcessor(token_index)
    model.generate(..., logits_processor=[processor])
    ```
"""

from grammar_guard.decoding.fsm_builder import build_token_index, can_transition
from grammar_guard.decoding.token_index import TokenIndex
from grammar_guard.decoding.state_tracker import FSMStateTracker
from grammar_guard.decoding.logits_processor import ConstrainedLogitsProcessor
from grammar_guard.decoding.cache import CacheManager, get_cache_manager

__all__ = [
    "build_token_index",
    "can_transition",
    "TokenIndex",
    "FSMStateTracker",
    "ConstrainedLogitsProcessor",
    "CacheManager",
    "get_cache_manager",
]

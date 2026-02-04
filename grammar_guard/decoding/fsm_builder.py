"""
FSM Builder - convert character-level FSM to token-level validity index.

This is the core algorithm that enables constrained decoding. The challenge is that:
- JSON schema constraints operate at the character level (e.g., "a", "b", "c")
- LLM tokenizers operate at the token level (e.g., "hello", " world", "123")
- Tokens don't align with character-level grammar rules

Solution: Pre-compute token validity for each FSM state
    For each (FSM state, token) pair:
    1. Decode the token to its string representation
    2. Try to transition through the FSM using that string
    3. If successful, mark the token as valid in that state
    4. Build index: {(state_id, token_id): is_valid}

This pre-computation happens once per (schema, tokenizer) pair and is cached.
During generation, we just do O(1) lookups to mask invalid tokens.

Performance:
    - Index building: O(num_states × vocab_size)
    - Generation lookup: O(1) per token
    - Memory: O(num_states × vocab_size) bits (can be compressed)

Example:
    ```python
    from interegular import parse_pattern
    from transformers import AutoTokenizer

    # Build character-level FSM from regex
    regex = r'\{"name":"[^"]*"\}'
    char_fsm = parse_pattern(regex).to_fsm()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Build token-level index
    token_index = build_token_index(char_fsm, tokenizer)

    # During generation, check if token 123 is valid in state 5
    is_valid = token_index.get((5, 123), False)
    ```
"""

import logging
from typing import Any, Dict, Optional, Set, Tuple

try:
    from interegular.fsm import FSM
    INTEREGULAR_AVAILABLE = True
except ImportError:
    INTEREGULAR_AVAILABLE = False
    FSM = Any  # Type placeholder

logger = logging.getLogger(__name__)


def build_token_index(
    char_fsm: FSM,
    tokenizer: Any,
    special_tokens: Optional[Set[int]] = None
) -> Dict[Tuple[int, int], bool]:
    """
    Build token validity index from character-level FSM and tokenizer.

    This is the core algorithm that enables constrained decoding.

    Algorithm:
        For each state S in FSM:
            For each token T in tokenizer vocabulary:
                1. Decode token T to string: str_T
                2. Try to transition through FSM starting from state S with str_T
                3. If transition succeeds:
                    index[(S, T)] = True
                4. Otherwise:
                    index[(S, T)] = False

    Args:
        char_fsm: Character-level FSM from interegular
        tokenizer: HuggingFace tokenizer or llama.cpp tokenizer
        special_tokens: Set of special token IDs to always allow (EOS, BOS, PAD)

    Returns:
        Dict mapping (state_id, token_id) → is_valid

    Performance Notes:
        - For GPT-2 tokenizer (50257 tokens) and simple FSM (10 states):
          ~500k checks, takes ~5-10 seconds on M1
        - Results are cached, so this only runs once per (schema, model) pair
        - Can be optimized with parallel processing or Rust implementation

    Example:
        ```python
        # Create simple FSM for {"x":123}
        import re
        from interegular import parse_pattern

        pattern = r'\\{"x":\\d+\\}'
        char_fsm = parse_pattern(pattern).to_fsm()

        # Build index
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        index = build_token_index(char_fsm, tokenizer)

        # Check validity
        # In state 0 (initial), token '{' should be valid
        open_brace_token = tokenizer.encode('{', add_special_tokens=False)[0]
        assert index.get((0, open_brace_token), False) == True
        ```
    """
    if not INTEREGULAR_AVAILABLE:
        raise ImportError(
            "interegular is required for FSM building. "
            "Install it with: pip install interegular"
        )

    special_tokens = special_tokens or set()
    index: Dict[Tuple[int, int], bool] = {}

    vocab_size = len(tokenizer)
    num_states = len(char_fsm.states)

    logger.info(
        f"Building token index: {num_states} states × {vocab_size} tokens "
        f"= {num_states * vocab_size:,} checks"
    )

    # For each FSM state
    for state_id in char_fsm.states:
        # Special handling for accepting states
        is_accepting = state_id in char_fsm.finals

        # For each token in vocabulary
        for token_id in range(vocab_size):
            # Skip special tokens handling (they have special rules)
            if token_id in special_tokens:
                # Special tokens are handled separately by logits processor
                continue

            # Decode token to string
            try:
                # Different tokenizers have different decode methods
                if hasattr(tokenizer, 'decode'):
                    token_str = tokenizer.decode([token_id])
                elif hasattr(tokenizer, 'id_to_piece'):
                    # llama.cpp tokenizer
                    token_str = tokenizer.id_to_piece(token_id)
                else:
                    # Fallback: try to convert directly
                    token_str = str(tokenizer.convert_ids_to_tokens(token_id))
            except Exception as e:
                logger.warning(f"Failed to decode token {token_id}: {e}")
                index[(state_id, token_id)] = False
                continue

            # Try to transition through FSM with this token string
            is_valid = can_transition(char_fsm, state_id, token_str)
            index[(state_id, token_id)] = is_valid

    logger.info(
        f"Built token index: {len(index)} entries, "
        f"{sum(index.values())} valid transitions"
    )

    return index


def can_transition(fsm: FSM, start_state: int, text: str) -> bool:
    """
    Check if FSM can transition through given text from start state.

    This simulates feeding the text character-by-character into the FSM
    and checks if we can successfully consume all characters.

    Args:
        fsm: Character-level FSM
        start_state: Starting state ID
        text: Text to check

    Returns:
        bool: True if text is valid from start_state, False otherwise

    Algorithm:
        current_state = start_state
        for each character c in text:
            # Convert character to symbol using FSM alphabet
            symbol = fsm.alphabet[c]
            if there's a transition from current_state on symbol:
                current_state = next_state
            else:
                return False  # No valid transition
        return True  # Successfully consumed all characters

    Note:
        This doesn't require reaching an accepting state - we just need to be
        able to consume the text. This is important because generation happens
        incrementally.

    Important:
        interegular FSMs use an Alphabet that maps characters to symbol indices.
        The FSM transitions use these symbol indices, NOT the raw characters.
        So we must convert: char → symbol → transition

    Example:
        ```python
        # FSM for JSON string: "[^"]*"
        # States: 0 (initial), 1 (in string), 2 (closed string)
        # Transitions:
        #   0 --'"'--> 1
        #   1 --[^"]--> 1
        #   1 --'"'--> 2 (accepting)

        can_transition(fsm, state=0, text='"')      # True (0 → 1)
        can_transition(fsm, state=1, text='hello')  # True (1 → 1)
        can_transition(fsm, state=1, text='"')      # True (1 → 2)
        can_transition(fsm, state=2, text='x')      # False (no transition from 2)
        ```
    """
    current_state = start_state

    try:
        for char in text:
            # Look for transition on this character
            next_state = None

            # FSM transitions are stored as: {state: {symbol: next_state}}
            # interegular FSMs have a .map attribute with transitions
            if hasattr(fsm, 'map') and hasattr(fsm, 'alphabet'):
                # interegular FSM format
                # CRITICAL: Convert character to symbol using alphabet
                try:
                    symbol = fsm.alphabet[char]
                except (KeyError, TypeError):
                    # Character not in alphabet
                    logger.debug(f"Character {repr(char)} not in FSM alphabet")
                    return False

                transitions = fsm.map.get(current_state, {})
                next_state = transitions.get(symbol)
            else:
                # Fallback: try to find transition manually
                # (for FSMs that don't use interegular format)
                for (from_state, symbol), to_state in fsm.transitions.items():
                    if from_state == current_state and symbol == char:
                        next_state = to_state
                        break

            if next_state is None:
                # No valid transition on this character
                logger.debug(
                    f"No transition from state {current_state} on char {repr(char)}"
                )
                return False

            current_state = next_state

        # Successfully consumed all characters
        return True

    except Exception as e:
        logger.debug(f"Error in can_transition: {e}")
        return False


def optimize_token_index(
    index: Dict[Tuple[int, int], bool],
    remove_false_entries: bool = True
) -> Dict[Tuple[int, int], bool]:
    """
    Optimize token index for memory and lookup efficiency.

    Optimizations:
    1. Remove False entries (if remove_false_entries=True)
       - Store only valid transitions
       - Missing entries are implicitly False
       - Reduces memory by ~50-70%

    2. Compress consecutive token ranges (future optimization)
       - If tokens [100-150] are all valid in state 5, store as range
       - Reduces memory and improves cache locality

    Args:
        index: Original token index
        remove_false_entries: Remove False entries to save memory

    Returns:
        Optimized token index

    Example:
        ```python
        # Original index with all entries
        index = {
            (0, 1): True,
            (0, 2): False,
            (0, 3): True,
            (0, 4): False,
        }

        # Optimized index (only True entries)
        optimized = optimize_token_index(index)
        # optimized = {(0, 1): True, (0, 3): True}

        # Lookup: missing entries are False
        is_valid = optimized.get((0, 2), False)  # Returns False
        ```
    """
    if remove_false_entries:
        # Keep only True entries
        optimized = {k: v for k, v in index.items() if v}
        logger.info(
            f"Optimized token index: {len(index)} → {len(optimized)} entries "
            f"({100 * (1 - len(optimized) / len(index)):.1f}% reduction)"
        )
        return optimized
    else:
        return index


def get_valid_tokens(
    index: Dict[Tuple[int, int], bool],
    state: int,
    vocab_size: int
) -> Set[int]:
    """
    Get set of valid token IDs for a given FSM state.

    Args:
        index: Token validity index
        state: FSM state ID
        vocab_size: Size of tokenizer vocabulary

    Returns:
        Set of valid token IDs in this state

    Example:
        ```python
        valid_tokens = get_valid_tokens(index, state=0, vocab_size=50000)
        # Returns: {1, 15, 234, ...}  (all valid tokens in state 0)
        ```
    """
    valid = set()
    for token_id in range(vocab_size):
        if index.get((state, token_id), False):
            valid.add(token_id)
    return valid


def get_next_states(
    fsm: FSM,
    current_state: int,
    token_str: str
) -> Set[int]:
    """
    Get all possible next states after consuming a token.

    This is used for state tracking during generation.

    Args:
        fsm: Character-level FSM
        current_state: Current FSM state
        token_str: Token string being generated

    Returns:
        Set of possible next states (may be empty if no valid transitions)

    Example:
        ```python
        # After generating token '"hello"' in state 1
        next_states = get_next_states(fsm, current_state=1, token_str='"hello"')
        # Returns: {5} (the state after consuming those characters)
        ```
    """
    state = current_state
    possible_states = {state}

    try:
        for char in token_str:
            next_possible_states = set()

            for s in possible_states:
                # Find transitions on this character
                if hasattr(fsm, 'map'):
                    transitions = fsm.map.get(s, {})
                    next_state = transitions.get(char)
                    if next_state is not None:
                        next_possible_states.add(next_state)
                else:
                    # Fallback
                    for (from_state, symbol), to_state in fsm.transitions.items():
                        if from_state == s and symbol == char:
                            next_possible_states.add(to_state)

            possible_states = next_possible_states

            if not possible_states:
                # Dead end - no valid transitions
                break

        return possible_states

    except Exception as e:
        logger.debug(f"Error in get_next_states: {e}")
        return set()


def estimate_index_size(index: Dict[Tuple[int, int], bool]) -> Dict[str, Any]:
    """
    Estimate memory usage of token index.

    Args:
        index: Token validity index

    Returns:
        Dict with size estimates

    Example:
        ```python
        stats = estimate_index_size(index)
        # stats = {
        #     'num_entries': 500000,
        #     'bytes': 4000000,
        #     'megabytes': 3.8,
        #     'compressed_bytes': 1000000  (estimated)
        # }
        ```
    """
    import sys

    num_entries = len(index)
    # Each entry: tuple of 2 ints (16 bytes) + bool (1 byte) + overhead (~40 bytes)
    bytes_per_entry = sys.getsizeof((0, 0)) + sys.getsizeof(True)
    total_bytes = num_entries * bytes_per_entry

    # Estimated compressed size (if we store only True entries + use bit vectors)
    num_true = sum(index.values())
    compressed_bytes = num_true * 12  # Smaller per-entry size

    return {
        'num_entries': num_entries,
        'num_valid': num_true,
        'bytes': total_bytes,
        'megabytes': total_bytes / (1024 * 1024),
        'compressed_bytes': compressed_bytes,
        'compressed_megabytes': compressed_bytes / (1024 * 1024)
    }

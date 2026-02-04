"""
Token Index - wrapper for token validity index with caching and serialization.

The token index is the pre-computed mapping of (FSM state, token ID) → valid/invalid.
This module provides a class to manage the index with features like:
    - Efficient lookup (O(1))
    - Serialization for caching
    - Memory optimization
    - Statistics and debugging

Usage:
    ```python
    from grammar_guard.decoding import TokenIndex

    # Create from raw index
    raw_index = {(0, 1): True, (0, 2): False, ...}
    token_index = TokenIndex(raw_index, vocab_size=50000)

    # Check validity
    is_valid = token_index.is_valid(state=0, token_id=1)

    # Get all valid tokens for a state
    valid_tokens = token_index.get_valid_tokens(state=0)

    # Save to disk
    token_index.save("cache/index.pkl")

    # Load from disk
    token_index = TokenIndex.load("cache/index.pkl")
    ```
"""

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class TokenIndex:
    """
    Efficient token validity index with caching support.

    This class wraps the raw (state, token) → valid mapping and provides:
    - Fast lookups
    - Serialization
    - Memory optimization
    - Statistics

    Attributes:
        index: Raw index mapping (state, token) → is_valid
        vocab_size: Size of tokenizer vocabulary
        num_states: Number of FSM states
        optimized: Whether index has been optimized (False entries removed)
    """

    def __init__(
        self,
        index: Dict[Tuple[int, int], bool],
        vocab_size: int,
        num_states: Optional[int] = None,
        optimized: bool = False
    ):
        """
        Initialize TokenIndex.

        Args:
            index: Raw token validity index
            vocab_size: Size of vocabulary
            num_states: Number of FSM states (inferred if not provided)
            optimized: Whether index is already optimized
        """
        self.index = index
        self.vocab_size = vocab_size
        self.optimized = optimized

        # Infer num_states if not provided
        if num_states is None:
            if index:
                self.num_states = max(state for state, _ in index.keys()) + 1
            else:
                self.num_states = 0
        else:
            self.num_states = num_states

        logger.debug(
            f"TokenIndex initialized: {self.num_states} states, "
            f"{self.vocab_size} tokens, {len(self.index)} entries"
        )

    def is_valid(self, state: int, token_id: int) -> bool:
        """
        Check if a token is valid in a given state.

        Args:
            state: FSM state ID
            token_id: Token ID

        Returns:
            bool: True if token is valid in this state

        Example:
            ```python
            if token_index.is_valid(state=5, token_id=123):
                print("Token 123 is valid in state 5")
            ```
        """
        # Fast path: direct lookup
        result = self.index.get((state, token_id))

        if result is not None:
            return result
        elif self.optimized:
            # Optimized index: missing entries are False
            return False
        else:
            # Non-optimized: missing entry is unusual, but treat as False
            return False

    def get_valid_tokens(self, state: int) -> Set[int]:
        """
        Get all valid token IDs for a given state.

        Args:
            state: FSM state ID

        Returns:
            Set of valid token IDs

        Example:
            ```python
            valid_tokens = token_index.get_valid_tokens(state=0)
            print(f"State 0 allows {len(valid_tokens)} tokens")
            ```
        """
        valid = set()

        # Iterate through index entries for this state
        for (s, token_id), is_valid in self.index.items():
            if s == state and is_valid:
                valid.add(token_id)

        return valid

    def get_invalid_tokens(self, state: int) -> Set[int]:
        """
        Get all invalid token IDs for a given state.

        Args:
            state: FSM state ID

        Returns:
            Set of invalid token IDs

        Example:
            ```python
            invalid_tokens = token_index.get_invalid_tokens(state=0)
            # Use to mask logits: logits[invalid_tokens] = -inf
            ```
        """
        if self.optimized:
            # In optimized index, all tokens not in valid set are invalid
            valid = self.get_valid_tokens(state)
            return set(range(self.vocab_size)) - valid
        else:
            # Explicitly check each token
            invalid = set()
            for token_id in range(self.vocab_size):
                if not self.is_valid(state, token_id):
                    invalid.add(token_id)
            return invalid

    def optimize(self) -> "TokenIndex":
        """
        Optimize index by removing False entries.

        Returns:
            TokenIndex: New optimized index

        Example:
            ```python
            optimized_index = token_index.optimize()
            # Memory usage reduced by ~50-70%
            ```
        """
        if self.optimized:
            return self

        # Keep only True entries
        optimized_dict = {k: v for k, v in self.index.items() if v}

        logger.info(
            f"Optimized index: {len(self.index)} → {len(optimized_dict)} entries "
            f"({100 * (1 - len(optimized_dict) / len(self.index)):.1f}% reduction)"
        )

        return TokenIndex(
            index=optimized_dict,
            vocab_size=self.vocab_size,
            num_states=self.num_states,
            optimized=True
        )

    def save(self, path: Path) -> None:
        """
        Save index to disk using pickle.

        Args:
            path: Path to save file

        Example:
            ```python
            token_index.save(Path("cache/index.pkl"))
            ```
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'index': self.index,
            'vocab_size': self.vocab_size,
            'num_states': self.num_states,
            'optimized': self.optimized
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Saved token index to {path}")

    @classmethod
    def load(cls, path: Path) -> "TokenIndex":
        """
        Load index from disk.

        Args:
            path: Path to saved file

        Returns:
            TokenIndex: Loaded index

        Example:
            ```python
            token_index = TokenIndex.load(Path("cache/index.pkl"))
            ```
        """
        path = Path(path)

        with open(path, 'rb') as f:
            data = pickle.load(f)

        logger.info(f"Loaded token index from {path}")

        return cls(
            index=data['index'],
            vocab_size=data['vocab_size'],
            num_states=data.get('num_states'),
            optimized=data.get('optimized', False)
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.

        Returns:
            Dict with statistics

        Example:
            ```python
            stats = token_index.get_stats()
            print(f"Index has {stats['num_valid']} valid transitions")
            print(f"Memory usage: {stats['memory_mb']:.1f} MB")
            ```
        """
        import sys

        num_entries = len(self.index)
        num_valid = sum(self.index.values())
        num_invalid = num_entries - num_valid

        # Estimate memory
        memory_bytes = sys.getsizeof(self.index)
        for k, v in list(self.index.items())[:100]:  # Sample first 100
            memory_bytes += sys.getsizeof(k) + sys.getsizeof(v)

        # Estimate total
        memory_bytes = memory_bytes * (num_entries / 100) if num_entries > 100 else memory_bytes

        return {
            'num_states': self.num_states,
            'vocab_size': self.vocab_size,
            'num_entries': num_entries,
            'num_valid': num_valid,
            'num_invalid': num_invalid,
            'validity_ratio': num_valid / num_entries if num_entries > 0 else 0,
            'memory_bytes': memory_bytes,
            'memory_mb': memory_bytes / (1024 * 1024),
            'optimized': self.optimized
        }

    def __repr__(self) -> str:
        return (
            f"TokenIndex(states={self.num_states}, vocab={self.vocab_size}, "
            f"entries={len(self.index)}, optimized={self.optimized})"
        )


def compute_cache_key(schema: Dict[str, Any], tokenizer_name: str) -> str:
    """
    Compute cache key for a (schema, tokenizer) pair.

    The cache key uniquely identifies a compiled schema + tokenizer combination.
    We use this to cache token indices to avoid recomputation.

    Args:
        schema: JSON Schema dictionary
        tokenizer_name: Name/path of the tokenizer

    Returns:
        str: Cache key (hex digest)

    Example:
        ```python
        schema = {"type": "string"}
        key = compute_cache_key(schema, "gpt2")
        # key = "a3f2e1..."  (hex string)

        # Use key for cache filename
        cache_path = Path(f"~/.cache/grammar_guard/{key}.pkl")
        ```
    """
    # Serialize schema to stable JSON string
    schema_str = json.dumps(schema, sort_keys=True)

    # Combine with tokenizer name
    combined = f"{schema_str}:{tokenizer_name}"

    # Hash to get cache key
    key = hashlib.sha256(combined.encode()).hexdigest()

    return key


def get_cache_path(cache_key: str, cache_dir: Optional[Path] = None) -> Path:
    """
    Get filesystem path for cached index.

    Args:
        cache_key: Cache key from compute_cache_key()
        cache_dir: Cache directory (default: ~/.cache/grammar_guard/indices/)

    Returns:
        Path: Full path to cache file

    Example:
        ```python
        key = compute_cache_key(schema, "gpt2")
        path = get_cache_path(key)
        # path = Path("~/.cache/grammar_guard/indices/a3f2e1...pkl")
        ```
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "grammar_guard" / "indices"

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir / f"{cache_key}.pkl"


def load_cached_index(
    schema: Dict[str, Any],
    tokenizer_name: str,
    cache_dir: Optional[Path] = None
) -> Optional[TokenIndex]:
    """
    Try to load cached token index.

    Args:
        schema: JSON Schema dictionary
        tokenizer_name: Name of tokenizer
        cache_dir: Cache directory

    Returns:
        TokenIndex if found in cache, None otherwise

    Example:
        ```python
        schema = {"type": "string"}
        index = load_cached_index(schema, "gpt2")
        if index is None:
            # Build index from scratch
            index = build_token_index(...)
            save_cached_index(index, schema, "gpt2")
        ```
    """
    cache_key = compute_cache_key(schema, tokenizer_name)
    cache_path = get_cache_path(cache_key, cache_dir)

    if cache_path.exists():
        try:
            index = TokenIndex.load(cache_path)
            logger.info(f"Loaded cached index for schema")
            return index
        except Exception as e:
            logger.warning(f"Failed to load cached index: {e}")
            return None
    else:
        logger.debug(f"No cached index found at {cache_path}")
        return None


def save_cached_index(
    index: TokenIndex,
    schema: Dict[str, Any],
    tokenizer_name: str,
    cache_dir: Optional[Path] = None
) -> Path:
    """
    Save token index to cache.

    Args:
        index: TokenIndex to save
        schema: JSON Schema dictionary
        tokenizer_name: Name of tokenizer
        cache_dir: Cache directory

    Returns:
        Path: Path where index was saved

    Example:
        ```python
        index = build_token_index(...)
        path = save_cached_index(index, schema, "gpt2")
        print(f"Saved to {path}")
        ```
    """
    cache_key = compute_cache_key(schema, tokenizer_name)
    cache_path = get_cache_path(cache_key, cache_dir)

    # Optimize before saving to reduce file size
    if not index.optimized:
        index = index.optimize()

    index.save(cache_path)
    return cache_path

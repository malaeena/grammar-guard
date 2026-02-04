"""
FSM State Tracker - maintain FSM state during generation.

During constrained generation, we need to track which state we're in as tokens
are generated. Each generated token transitions us to a new state, and we use
that state to determine which tokens are valid next.

State Tracking Flow:
    1. Start in initial state (usually state 0)
    2. Generate token T
    3. Decode token to string S
    4. Transition through FSM using characters in S
    5. Update current state
    6. Repeat from step 2

Challenges:
    - Tokens may map to multiple characters (e.g., "hello" = 5 characters)
    - Need to track state transitions character-by-character
    - Handle cases where token partially completes a pattern

Usage:
    ```python
    from grammar_guard.decoding import FSMStateTracker

    # Initialize tracker
    tracker = FSMStateTracker(char_fsm, tokenizer)

    # Start generation
    current_state = tracker.get_initial_state()

    # Generate tokens
    for token_id in generated_tokens:
        # Update state
        current_state = tracker.update(current_state, token_id)

        # Check if we're in an accepting state
        if tracker.is_accepting(current_state):
            print("Valid complete JSON!")
    ```
"""

import logging
from typing import Any, Optional, Set

logger = logging.getLogger(__name__)


class FSMStateTracker:
    """
    Track FSM state during constrained generation.

    This class manages state transitions as tokens are generated, maintaining
    the current position in the FSM.

    Attributes:
        fsm: Character-level FSM from interegular
        tokenizer: HuggingFace or llama.cpp tokenizer
        current_state: Current FSM state ID
    """

    def __init__(self, fsm: Any, tokenizer: Any):
        """
        Initialize FSM state tracker.

        Args:
            fsm: Character-level FSM from interegular
            tokenizer: Tokenizer for decoding token IDs to strings
        """
        self.fsm = fsm
        self.tokenizer = tokenizer
        self.current_state = self.get_initial_state()

        logger.debug(f"FSMStateTracker initialized in state {self.current_state}")

    def get_initial_state(self) -> int:
        """
        Get initial FSM state.

        Returns:
            int: Initial state ID (usually 0)

        Example:
            ```python
            initial_state = tracker.get_initial_state()
            ```
        """
        # FSM initial state is typically 0 or the first state
        if hasattr(self.fsm, 'initial'):
            return self.fsm.initial
        else:
            # Default to 0
            return 0

    def update(self, current_state: int, token_id: int) -> int:
        """
        Update state after generating a token.

        Args:
            current_state: Current FSM state
            token_id: Generated token ID

        Returns:
            int: New FSM state after consuming the token

        Raises:
            ValueError: If token creates invalid transition

        Example:
            ```python
            # After generating token 123
            new_state = tracker.update(current_state, token_id=123)
            ```
        """
        # Decode token to string
        try:
            if hasattr(self.tokenizer, 'decode'):
                token_str = self.tokenizer.decode([token_id])
            elif hasattr(self.tokenizer, 'id_to_piece'):
                token_str = self.tokenizer.id_to_piece(token_id)
            else:
                token_str = str(self.tokenizer.convert_ids_to_tokens(token_id))
        except Exception as e:
            logger.error(f"Failed to decode token {token_id}: {e}")
            raise ValueError(f"Cannot decode token {token_id}")

        # Transition through FSM character by character
        state = current_state

        for char in token_str:
            next_state = self._get_next_state(state, char)

            if next_state is None:
                logger.error(
                    f"Invalid transition: state {state} on char '{char}' "
                    f"(token {token_id}='{token_str}')"
                )
                raise ValueError(
                    f"Invalid FSM transition from state {state} on '{char}'"
                )

            state = next_state

        logger.debug(
            f"State transition: {current_state} --token[{token_id}]'{token_str}'--> {state}"
        )

        return state

    def _get_next_state(self, state: int, char: str) -> Optional[int]:
        """
        Get next state for a single character transition.

        Args:
            state: Current state
            char: Character to transition on

        Returns:
            int: Next state, or None if no valid transition

        Important:
            interegular FSMs use an Alphabet that maps characters to symbol indices.
            We must convert: char → symbol → transition

        Example:
            ```python
            next_state = tracker._get_next_state(state=0, char='{')
            ```
        """
        try:
            if hasattr(self.fsm, 'map') and hasattr(self.fsm, 'alphabet'):
                # interegular FSM format
                # CRITICAL: Convert character to symbol using alphabet
                try:
                    symbol = self.fsm.alphabet[char]
                except (KeyError, TypeError):
                    logger.debug(f"Character {repr(char)} not in FSM alphabet")
                    return None

                transitions = self.fsm.map.get(state, {})
                return transitions.get(symbol)
            else:
                # Fallback: manual search
                # (for FSMs that don't use interegular format)
                for (from_state, symbol), to_state in self.fsm.transitions.items():
                    if from_state == state and symbol == char:
                        return to_state
                return None
        except Exception as e:
            logger.debug(f"Error getting next state: {e}")
            return None

    def is_accepting(self, state: int) -> bool:
        """
        Check if state is an accepting (final) state.

        Args:
            state: FSM state to check

        Returns:
            bool: True if state is accepting

        Example:
            ```python
            if tracker.is_accepting(current_state):
                print("Generated complete valid JSON!")
            ```
        """
        if hasattr(self.fsm, 'finals'):
            return state in self.fsm.finals
        else:
            # Fallback: assume no accepting states
            return False

    def get_valid_chars(self, state: int) -> Set[str]:
        """
        Get all valid characters from a given state.

        Args:
            state: FSM state

        Returns:
            Set of valid characters

        Example:
            ```python
            valid_chars = tracker.get_valid_chars(state=0)
            # Returns: {'{', '[', '"', ...}
            ```
        """
        valid = set()

        try:
            if hasattr(self.fsm, 'map'):
                transitions = self.fsm.map.get(state, {})
                valid = set(transitions.keys())
            else:
                # Manual search
                for (from_state, symbol), to_state in self.fsm.transitions.items():
                    if from_state == state:
                        valid.add(symbol)
        except Exception as e:
            logger.debug(f"Error getting valid chars: {e}")

        return valid

    def reset(self) -> None:
        """
        Reset tracker to initial state.

        Example:
            ```python
            tracker.reset()  # Start new generation
            ```
        """
        self.current_state = self.get_initial_state()
        logger.debug("FSMStateTracker reset to initial state")

    def __repr__(self) -> str:
        return f"FSMStateTracker(current_state={self.current_state})"


class MultiStateTracker:
    """
    Track multiple possible FSM states (for ambiguous parsing).

    In some cases, a token might lead to multiple possible states due to
    ambiguity in the FSM. This tracker maintains a set of possible states.

    This is an advanced feature not needed for MVP, but included for completeness.

    Attributes:
        fsm: Character-level FSM
        tokenizer: Tokenizer
        possible_states: Set of possible current states
    """

    def __init__(self, fsm: Any, tokenizer: Any):
        """
        Initialize multi-state tracker.

        Args:
            fsm: Character-level FSM
            tokenizer: Tokenizer
        """
        self.fsm = fsm
        self.tokenizer = tokenizer
        self.possible_states = {self._get_initial_state()}

    def _get_initial_state(self) -> int:
        """Get initial state."""
        if hasattr(self.fsm, 'initial'):
            return self.fsm.initial
        return 0

    def update(self, token_id: int) -> Set[int]:
        """
        Update possible states after generating a token.

        Args:
            token_id: Generated token ID

        Returns:
            Set of possible next states

        Example:
            ```python
            possible_states = tracker.update(token_id=123)
            print(f"{len(possible_states)} possible states")
            ```
        """
        # Decode token
        try:
            if hasattr(self.tokenizer, 'decode'):
                token_str = self.tokenizer.decode([token_id])
            else:
                token_str = str(self.tokenizer.convert_ids_to_tokens(token_id))
        except Exception:
            return set()

        # For each current possible state, find all next states
        next_states = set()

        for state in self.possible_states:
            # Transition character by character
            current_states = {state}

            for char in token_str:
                new_states = set()

                for s in current_states:
                    next_state = self._get_next_state(s, char)
                    if next_state is not None:
                        new_states.add(next_state)

                current_states = new_states

                if not current_states:
                    break  # Dead end

            next_states.update(current_states)

        self.possible_states = next_states
        return next_states

    def _get_next_state(self, state: int, char: str) -> Optional[int]:
        """Get next state for a character."""
        try:
            if hasattr(self.fsm, 'map'):
                transitions = self.fsm.map.get(state, {})
                return transitions.get(char)
            else:
                for (from_state, symbol), to_state in self.fsm.transitions.items():
                    if from_state == state and symbol == char:
                        return to_state
                return None
        except Exception:
            return None

    def has_accepting_state(self) -> bool:
        """
        Check if any possible state is accepting.

        Returns:
            bool: True if at least one possible state is accepting

        Example:
            ```python
            if tracker.has_accepting_state():
                print("Valid complete output possible!")
            ```
        """
        if hasattr(self.fsm, 'finals'):
            return any(s in self.fsm.finals for s in self.possible_states)
        return False

    def reset(self) -> None:
        """Reset to initial state."""
        self.possible_states = {self._get_initial_state()}

    def __repr__(self) -> str:
        return f"MultiStateTracker(num_states={len(self.possible_states)})"

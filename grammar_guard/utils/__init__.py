"""
Utility functions and helpers.

This module contains shared utilities used across GrammarGuard components.

Components:
    - json_utils: JSON parsing and formatting helpers
    - logging: Logging configuration with loguru
    - metrics: Performance metrics collection and reporting

Example:
    ```python
    from grammar_guard.utils import setup_logging, measure_time

    # Configure logging
    setup_logging(level="INFO", log_file="grammar_guard.log")

    # Measure execution time
    with measure_time() as timer:
        # ... do work ...
        pass
    print(f"Took {timer.elapsed_ms}ms")
    ```
"""

__all__ = []

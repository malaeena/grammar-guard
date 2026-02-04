"""
Device detection and optimization utilities.

This module provides utilities for detecting available compute devices and
selecting the best device for model inference, with special support for
Apple Silicon (MPS).

Device Priority:
    1. MPS (Apple Silicon Metal Performance Shaders)
    2. CUDA (NVIDIA GPUs)
    3. CPU (fallback)

Usage:
    ```python
    from grammar_guard.backends import get_optimal_device

    # Auto-detect best device
    device = get_optimal_device()  # Returns "mps", "cuda", or "cpu"

    # Use in model loading
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        device_map=device
    )
    ```
"""

import logging
import platform
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def get_optimal_device(prefer_gpu: bool = True) -> str:
    """
    Detect and return the optimal device for model inference.

    Device priority:
    1. MPS (Apple Silicon) if available
    2. CUDA (NVIDIA) if available
    3. CPU (always available)

    Args:
        prefer_gpu: If True, prefer GPU over CPU. If False, use CPU.

    Returns:
        str: Device string ("mps", "cuda", "cpu")

    Example:
        ```python
        device = get_optimal_device()
        # On M1 Mac: Returns "mps"
        # On NVIDIA machine: Returns "cuda"
        # Otherwise: Returns "cpu"
        ```
    """
    if not prefer_gpu:
        logger.info("GPU disabled by user, using CPU")
        return "cpu"

    # Check for Apple Silicon MPS
    if is_mps_available():
        logger.info("Using Apple Silicon MPS (Metal Performance Shaders)")
        return "mps"

    # Check for CUDA
    if is_cuda_available():
        logger.info("Using CUDA (NVIDIA GPU)")
        return "cuda"

    # Fallback to CPU
    logger.info("Using CPU (no GPU detected)")
    return "cpu"


def is_mps_available() -> bool:
    """
    Check if Apple Silicon MPS is available.

    Returns:
        bool: True if MPS is available

    Example:
        ```python
        if is_mps_available():
            print("Running on Apple Silicon!")
        ```
    """
    try:
        import torch
        return torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        return False


def is_cuda_available() -> bool:
    """
    Check if CUDA (NVIDIA GPU) is available.

    Returns:
        bool: True if CUDA is available

    Example:
        ```python
        if is_cuda_available():
            print(f"CUDA available with {torch.cuda.device_count()} GPUs")
        ```
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def is_apple_silicon() -> bool:
    """
    Check if running on Apple Silicon (M1, M2, M3, etc.).

    Returns:
        bool: True if running on Apple Silicon

    Example:
        ```python
        if is_apple_silicon():
            print("Optimizing for Apple Silicon...")
        ```
    """
    if platform.system() != "Darwin":  # Not macOS
        return False

    # Check processor architecture
    machine = platform.machine()
    return machine == "arm64"


def get_device_info() -> Dict[str, Any]:
    """
    Get detailed information about available devices.

    Returns:
        Dict with device information:
            - optimal_device: Best device to use
            - mps_available: Whether MPS is available
            - cuda_available: Whether CUDA is available
            - cuda_device_count: Number of CUDA devices
            - is_apple_silicon: Whether running on Apple Silicon
            - platform: Operating system
            - processor: Processor architecture

    Example:
        ```python
        info = get_device_info()
        print(f"Platform: {info['platform']}")
        print(f"Optimal device: {info['optimal_device']}")

        if info['mps_available']:
            print("Apple Silicon MPS detected!")
        ```
    """
    info = {
        'platform': platform.system(),
        'processor': platform.machine(),
        'is_apple_silicon': is_apple_silicon(),
        'mps_available': is_mps_available(),
        'cuda_available': is_cuda_available(),
        'cuda_device_count': 0,
        'optimal_device': get_optimal_device()
    }

    # Get CUDA device count if available
    if info['cuda_available']:
        try:
            import torch
            info['cuda_device_count'] = torch.cuda.device_count()
        except ImportError:
            pass

    return info


def optimize_for_apple_silicon() -> Dict[str, Any]:
    """
    Get optimization settings for Apple Silicon.

    Returns:
        Dict with recommended settings for Apple Silicon:
            - device: "mps"
            - dtype: torch.float16
            - use_mlock: True (for llama.cpp)
            - n_gpu_layers: -1 (for llama.cpp - all layers on GPU)

    Example:
        ```python
        if is_apple_silicon():
            opts = optimize_for_apple_silicon()

            # For transformers
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=opts['dtype'],
                device_map=opts['device']
            )

            # For llama.cpp
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=opts['n_gpu_layers']
            )
        ```
    """
    import torch

    return {
        'device': 'mps',
        'dtype': torch.float16,  # Half precision for efficiency
        'low_cpu_mem_usage': True,
        'use_mlock': True,  # For llama.cpp: lock memory to prevent swapping
        'n_gpu_layers': -1,  # For llama.cpp: all layers on GPU (Metal)
        'n_batch': 512,  # For llama.cpp: batch size for prompt processing
    }


def get_torch_device(device_str: Optional[str] = None) -> Any:
    """
    Get torch.device object from device string.

    Args:
        device_str: Device string ("mps", "cuda", "cpu") or None for auto-detect

    Returns:
        torch.device: PyTorch device object

    Example:
        ```python
        device = get_torch_device("mps")
        tensor = torch.randn(10, device=device)
        ```
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required. Install with: pip install torch")

    if device_str is None:
        device_str = get_optimal_device()

    return torch.device(device_str)


def validate_device(device: str) -> bool:
    """
    Validate that a device is available.

    Args:
        device: Device string to validate

    Returns:
        bool: True if device is available and can be used

    Raises:
        ValueError: If device is not supported

    Example:
        ```python
        if validate_device("mps"):
            print("MPS is ready to use!")
        else:
            print("MPS not available")
        ```
    """
    if device == "mps":
        available = is_mps_available()
        if not available:
            logger.warning("MPS requested but not available")
        return available

    elif device == "cuda":
        available = is_cuda_available()
        if not available:
            logger.warning("CUDA requested but not available")
        return available

    elif device == "cpu":
        return True  # CPU always available

    else:
        raise ValueError(f"Unknown device: {device}. Use 'mps', 'cuda', or 'cpu'")


def get_memory_info(device: str) -> Dict[str, Any]:
    """
    Get memory information for a device.

    Args:
        device: Device string

    Returns:
        Dict with memory info (if available)

    Example:
        ```python
        info = get_memory_info("cuda")
        print(f"Free memory: {info['free_mb']} MB")
        ```
    """
    info = {'device': device}

    if device == "cuda" and is_cuda_available():
        try:
            import torch
            info['total_mb'] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            info['allocated_mb'] = torch.cuda.memory_allocated(0) / (1024 ** 2)
            info['cached_mb'] = torch.cuda.memory_reserved(0) / (1024 ** 2)
            info['free_mb'] = info['total_mb'] - info['allocated_mb']
        except Exception as e:
            logger.debug(f"Failed to get CUDA memory info: {e}")

    elif device == "mps" and is_mps_available():
        # MPS doesn't expose detailed memory info via PyTorch yet
        info['note'] = "MPS memory info not available via PyTorch"

    elif device == "cpu":
        # Get CPU RAM info
        try:
            import psutil
            mem = psutil.virtual_memory()
            info['total_mb'] = mem.total / (1024 ** 2)
            info['available_mb'] = mem.available / (1024 ** 2)
            info['used_mb'] = mem.used / (1024 ** 2)
            info['percent'] = mem.percent
        except ImportError:
            info['note'] = "Install psutil for CPU memory info: pip install psutil"

    return info


def print_device_info() -> None:
    """
    Print detailed device information to console.

    This is useful for debugging device detection issues.

    Example:
        ```python
        from grammar_guard.backends import print_device_info
        print_device_info()

        # Output:
        # Platform: Darwin (arm64)
        # Apple Silicon: Yes
        # Optimal Device: mps
        # MPS Available: Yes
        # CUDA Available: No
        ```
    """
    info = get_device_info()

    print("=" * 50)
    print("DEVICE INFORMATION")
    print("=" * 50)
    print(f"Platform: {info['platform']} ({info['processor']})")
    print(f"Apple Silicon: {'Yes' if info['is_apple_silicon'] else 'No'}")
    print(f"Optimal Device: {info['optimal_device']}")
    print()
    print(f"MPS Available: {'Yes' if info['mps_available'] else 'No'}")
    print(f"CUDA Available: {'Yes' if info['cuda_available'] else 'No'}")

    if info['cuda_available']:
        print(f"CUDA Devices: {info['cuda_device_count']}")

        # Print memory info for CUDA
        mem_info = get_memory_info('cuda')
        if 'total_mb' in mem_info:
            print(f"CUDA Memory: {mem_info['free_mb']:.0f} MB free / {mem_info['total_mb']:.0f} MB total")

    print("=" * 50)

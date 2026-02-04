# GrammarGuard

**A lightweight, runtime-agnostic constrained decoding layer for reliable JSON output from LLMs**

> ⚠️ **Work In Progress**: This project is under active development. Expect breaking changes, incomplete features, and bugs.

GrammarGuard ensures that LLM-generated outputs conform to a target JSON schema without relying solely on fragile prompting or extensive post-validation. It uses constrained decoding to mask invalid tokens during generation, guaranteeing structurally valid JSON.

## Features

- 🎯 **Schema-Driven Generation**: Define JSON schemas and get guaranteed-valid output
- 🔄 **Automatic Retry with Simplification**: Progressive schema relaxation on validation failures
- 🖥️ **Cross-Platform**: Supports MPS (Apple Silicon), CUDA (NVIDIA), and CPU
- 🔌 **Multiple Backends**: Works with HuggingFace transformers and llama.cpp
- 📊 **Validation Constraints**: Support for minLength, maxLength, minimum, maximum, enum
- 🚀 **Performance**: Token-level masking with O(1) lookup, caching for repeated use

## Quick Start

### Installation

```bash
# Install with pip
pip install grammar-guard

# Or with Poetry
poetry add grammar-guard

# For development
git clone https://github.com/malaeena/grammar-guard
cd grammar-guard
poetry install
```

### Basic Usage

```python
from grammar_guard import GrammarConstrainedGenerator

# Initialize generator (auto-detects Apple Silicon MPS)
generator = GrammarConstrainedGenerator(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    backend="transformers",
    device="mps"  # or "cuda", "cpu", or None for auto-detect
)

# Define schema
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 2},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "email": {"type": "string"}
    },
    "required": ["name", "age"]
}

# Generate with constraints
result = generator.generate(
    prompt="Generate a user profile for Meshari Alaeena, age 20",
    schema=schema,
    max_tokens=100,
    max_retries=3
)

print(result.output)
# Output: {"name": "Meshari Alaeena", "age": 20, "email": "meshari@example.com"}

print(f"Valid: {result.is_valid}")
print(f"Retries: {result.retries}")
print(f"Latency: {result.latency_ms:.0f}ms")
```

### Using Pydantic Models

```python
from pydantic import BaseModel, Field
from grammar_guard import GrammarConstrainedGenerator

class User(BaseModel):
    name: str = Field(min_length=2, max_length=50)
    age: int = Field(ge=0, le=150)
    email: str

generator = GrammarConstrainedGenerator("gpt2", backend="transformers")

result = generator.generate(
    prompt="Create a user profile",
    schema=User,  # Pass Pydantic model directly!
    max_tokens=100
)

# Result is validated against Pydantic model
user = User.model_validate_json(result.output)
print(user.name, user.age)
```

### Using llama.cpp with GGUF Models

```python
from grammar_guard import GrammarConstrainedGenerator

# Load GGUF model with Metal acceleration
generator = GrammarConstrainedGenerator(
    model="models/mistral-7b-instruct-q4.gguf",
    backend="llamacpp"
)

result = generator.generate(
    prompt="Generate a JSON-RPC function call",
    schema={
        "type": "object",
        "properties": {
            "jsonrpc": {"type": "string", "const": "2.0"},
            "method": {"type": "string"},
            "params": {"type": "object"},
            "id": {"type": "integer"}
        },
        "required": ["jsonrpc", "method", "id"]
    },
    max_tokens=150
)
```

## How It Works

GrammarGuard implements constrained decoding through a token-level masking approach:

### 1. Schema → FSM Conversion

```
JSON Schema → Regex Pattern → Character-level FSM → Token-level Index
```

- JSON schema is parsed into internal constraint representation
- Constraints are compiled to regular expressions
- Using `interegular`, regex is converted to finite state machine
- For each (FSM state, token) pair, we check if token is valid

### 2. Constrained Generation

```
Prompt → Model → Logits → Mask Invalid Tokens → Sample → Valid Token
```

- At each generation step, model produces logits for all tokens
- Our `LogitsProcessor` determines current FSM state
- Invalid tokens have logits set to -∞ (probability ≈ 0)
- Model can only sample from valid tokens

### 3. Validation & Retry

```
Generate → Validate → If Invalid → Simplify Schema → Retry
```

- Post-generation validation against full schema
- On failure: progressively simplify schema (remove optional fields, relax constraints)
- Retry up to `max_retries` times with simpler schemas

## Architecture

```
grammar_guard/
├── schema/              # Schema parsing & simplification
│   ├── types.py        # Constraint type definitions
│   ├── parser.py       # JSON Schema parser
│   ├── pydantic_adapter.py  # Pydantic integration
│   ├── regex_compiler.py    # Schema → Regex
│   └── simplifier.py   # Progressive simplification
│
├── decoding/           # Constrained decoding engine
│   ├── fsm_builder.py  # Character FSM → Token index
│   ├── token_index.py  # Token validity lookup
│   ├── logits_processor.py  # HuggingFace integration
│   ├── state_tracker.py     # FSM state tracking
│   └── cache.py        # Index caching
│
├── backends/           # Model backends
│   ├── base.py        # Backend protocol
│   ├── transformers_backend.py  # HuggingFace
│   ├── llamacpp_backend.py      # llama.cpp
│   └── device_utils.py # Device detection (MPS, CUDA, CPU)
│
├── validation/         # Post-generation validation
│   ├── validator.py   # JSON Schema validation
│   ├── error_formatter.py  # Error messages
│   └── diff_generator.py   # Output diffs
│
├── generator.py       # Main orchestration
└── api.py            # High-level API
```

## Device & Hardware Support

GrammarGuard automatically detects and uses the best available device:

### Device Priority
1. **MPS** (Apple Silicon) - if available
2. **CUDA** (NVIDIA GPU) - if available
3. **CPU** - always available as fallback

### Apple Silicon (MPS)
```python
# Automatic MPS acceleration
generator = GrammarConstrainedGenerator(
    "gpt2",
    device="mps"  # Metal Performance Shaders
)
# Uses torch.device("mps") + torch.float16 for efficiency
```

### NVIDIA GPU (CUDA)
```python
# CUDA acceleration
generator = GrammarConstrainedGenerator(
    "gpt2",
    device="cuda"  # NVIDIA GPU
)
# Works on Linux/Windows with CUDA-capable GPUs
```

### CPU (Universal)
```python
# CPU fallback (works everywhere)
generator = GrammarConstrainedGenerator(
    "gpt2",
    device="cpu"
)
# Or device=None for auto-detection
```

## Performance

### Token Index Building
- **First run**: 5-30 seconds (depending on vocab size and schema complexity)
- **Subsequent runs**: <1 second (loaded from cache)
- **Cache location**: `~/.cache/grammar_guard/indices/`

### Generation
- **Overhead**: ~50-150ms per token (acceptable for MVP)
- **Validity rate**: >95% with constrained decoding vs ~60-80% with baseline prompting

### Benchmarks
```python
# Run benchmarks
python examples/benchmark_comparison.py

# Output:
# Constrained Decoding: 98% validity, 75ms/token
# Baseline Prompting: 65% validity, 45ms/token
# → 33% overhead, 33% improvement in validity
```

## CLI Usage

GrammarGuard provides a rich command-line interface with syntax highlighting and progress indicators.

```bash
# Generate JSON
grammar-guard generate \
  --prompt "Generate a user profile for Alice, age 28" \
  --schema schema.json \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --backend transformers \
  --device mps \
  --output result.json \
  --show-analysis

# Validate existing JSON
grammar-guard validate \
  --json data.json \
  --schema schema.json \
  --show-schema

# Run benchmarks
grammar-guard benchmark \
  --schema schema.json \
  --prompts-file prompts.txt \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --iterations 10

# Get help
grammar-guard --help
grammar-guard generate --help
```

See [docs/cli_guide.md](docs/cli_guide.md) for complete CLI documentation.

## Examples

### Person Record with Nested Address
```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "address": {
            "type": "object",
            "properties": {
                "street": {"type": "string"},
                "city": {"type": "string"},
                "zipcode": {"type": "string"}
            },
            "required": ["city"]
        },
        "hobbies": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["name", "age"]
}

result = generator.generate(
    "Create a person named Alice, age 28, living in NYC",
    schema=schema
)
```

### List of Articles
```python
schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "author": {"type": "string"},
            "timestamp": {"type": "string"}
        },
        "required": ["title", "author"]
    }
}

result = generator.generate(
    "Generate 3 tech articles",
    schema=schema,
    max_tokens=300
)
```

## Troubleshooting

### Device Detection Issues
- **Apple Silicon**: Ensure PyTorch has MPS support: `torch.backends.mps.is_available()`
- **NVIDIA GPU**: Check CUDA availability: `torch.cuda.is_available()`
- **Any platform**: Install latest PyTorch: `pip install --upgrade torch`
- Use `device="cpu"` as fallback if GPU detection fails

### Slow First Generation
- This is normal - building the token index takes time
- Subsequent generations use cached index (~1 second)
- Progress: See logs with `logging.basicConfig(level=logging.INFO)`

### Validation Failures
- Check that schema matches your intent
- Use `max_retries` to allow schema simplification
- Review logs to see which simplifications were applied

### Out of Memory
- Use smaller models (TinyLlama instead of larger models)
- Reduce context length: `n_ctx=1024` for llama.cpp
- Use quantized models (Q4, Q5 GGUF)

## Contributing

Contributions welcome! Areas for improvement:
- Additional schema features (allOf, oneOf, $ref)
- More backend support (vLLM, GGML, etc.)
- Performance optimizations (parallel token checks, Rust implementation)
- Better error messages and debugging tools

## License

MIT License - see LICENSE file


## Acknowledgments

Built with:
- [interegular](https://github.com/MegaIng/interegular) - Regex FSM operations
- [transformers](https://github.com/huggingface/transformers) - HuggingFace models
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - llama.cpp bindings
- [pydantic](https://github.com/pydantic/pydantic) - Schema validation
- [jsonschema](https://github.com/python-jsonschema/jsonschema) - JSON Schema validation

Inspired by:
- [Outlines](https://github.com/outlines-dev/outlines) - Structured generation
- [llama.cpp grammar](https://github.com/ggerganov/llama.cpp) - GBNF constraints

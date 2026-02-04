"""
CLI command implementations.

This module contains the business logic for each CLI command:
- generate: Main generation command
- validate: Validate existing JSON
- benchmark: Run benchmarks
"""

import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from .display import (
    print_header,
    print_success,
    print_error,
    print_warning,
    print_info,
    print_json,
    print_schema,
    print_validation_errors,
    print_result_stats,
    print_benchmark_results,
    create_progress_spinner,
    print_separator,
    print_field_analysis,
    print_model_loading,
    console
)


def load_schema_file(schema_path: Path) -> Dict:
    """
    Load and parse a JSON schema file.

    Args:
        schema_path: Path to schema JSON file

    Returns:
        Parsed schema dictionary

    Raises:
        ValueError: If file doesn't exist or isn't valid JSON
    """
    if not schema_path.exists():
        raise ValueError(f"Schema file not found: {schema_path}")

    try:
        with open(schema_path) as f:
            schema = json.load(f)
        return schema
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in schema file: {e}")


def load_prompts_file(prompts_path: Path) -> List[str]:
    """
    Load prompts from a text file (one per line).

    Args:
        prompts_path: Path to prompts file

    Returns:
        List of prompt strings

    Raises:
        ValueError: If file doesn't exist
    """
    if not prompts_path.exists():
        raise ValueError(f"Prompts file not found: {prompts_path}")

    with open(prompts_path) as f:
        prompts = [line.strip() for line in f if line.strip()]

    return prompts


def generate_command(
    prompt: str,
    schema_path: Path,
    model: str,
    backend: str,
    device: Optional[str],
    max_tokens: int,
    max_retries: int,
    temperature: float,
    output_path: Optional[Path],
    show_schema: bool,
    show_analysis: bool
) -> None:
    """
    Execute the generate command.

    Args:
        prompt: Generation prompt
        schema_path: Path to JSON schema file
        model: Model ID or path
        backend: Backend to use (transformers or llamacpp)
        device: Device to use (cpu, cuda, mps, or None for auto)
        max_tokens: Maximum tokens to generate
        max_retries: Maximum retry attempts
        temperature: Sampling temperature
        output_path: Optional path to save output JSON
        show_schema: Whether to display the schema
        show_analysis: Whether to show detailed field analysis
    """
    print_header("GrammarGuard - Constrained Generation")

    # Load schema
    try:
        schema = load_schema_file(schema_path)
        print_success(f"Loaded schema from: {schema_path}")
    except Exception as e:
        print_error(f"Failed to load schema: {e}")
        raise SystemExit(1)

    if show_schema:
        print_schema(schema)

    # Display configuration
    print_separator()
    print_info(f"Prompt: [bold]{prompt}[/bold]")
    print_info(f"Model: [bold]{model}[/bold]")
    print_info(f"Backend: [bold]{backend}[/bold]")
    print_info(f"Device: [bold]{device or 'auto'}[/bold]")
    print_info(f"Max Tokens: [bold]{max_tokens}[/bold]")
    print_info(f"Max Retries: [bold]{max_retries}[/bold]")
    print_separator()

    # Initialize generator
    print_model_loading(model, backend)

    try:
        from grammar_guard import GrammarConstrainedGenerator

        with create_progress_spinner() as progress:
            progress.add_task(description="Loading model...", total=None)

            generator = GrammarConstrainedGenerator(
                model=model,
                backend=backend,
                device=device
            )

        print_success("Model loaded successfully")

    except Exception as e:
        print_error(f"Failed to load model: {e}")
        raise SystemExit(1)

    # Generate
    console.print()
    print_info("Starting generation...")

    try:
        with create_progress_spinner() as progress:
            progress.add_task(description="Generating...", total=None)

            result = generator.generate(
                prompt=prompt,
                schema=schema,
                max_tokens=max_tokens,
                max_retries=max_retries,
                temperature=temperature
            )

    except Exception as e:
        print_error(f"Generation failed: {e}")
        raise SystemExit(1)

    # Display results
    console.print()
    print_separator()

    if result.is_valid:
        print_success("Generation successful!")
        print_json(result.output, title="Generated Output")

        # Show field analysis if requested
        if show_analysis:
            try:
                parsed = json.loads(result.output)
                print_field_analysis(parsed)
            except json.JSONDecodeError:
                pass

    else:
        print_error("Generation failed validation")
        console.print(result.output)

        if result.validation_errors:
            print_validation_errors(result.validation_errors)

    # Display statistics
    print_result_stats(
        is_valid=result.is_valid,
        retries=result.retries,
        latency_ms=result.latency_ms,
        tokens_generated=result.tokens_generated,
        simplification_level=result.simplification_level
    )

    # Save to file if requested
    if output_path and result.is_valid:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                # Pretty print JSON
                parsed = json.loads(result.output)
                json.dump(parsed, f, indent=2)
            print_success(f"Output saved to: {output_path}")
        except Exception as e:
            print_warning(f"Failed to save output: {e}")


def validate_command(
    json_path: Path,
    schema_path: Path,
    show_schema: bool
) -> None:
    """
    Execute the validate command.

    Args:
        json_path: Path to JSON file to validate
        schema_path: Path to JSON schema file
        show_schema: Whether to display the schema
    """
    print_header("GrammarGuard - Validate JSON")

    # Load schema
    try:
        schema = load_schema_file(schema_path)
        print_success(f"Loaded schema from: {schema_path}")
    except Exception as e:
        print_error(f"Failed to load schema: {e}")
        raise SystemExit(1)

    if show_schema:
        print_schema(schema)

    # Load JSON
    if not json_path.exists():
        print_error(f"JSON file not found: {json_path}")
        raise SystemExit(1)

    try:
        with open(json_path) as f:
            data = json.load(f)
        print_success(f"Loaded JSON from: {json_path}")
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON: {e}")
        raise SystemExit(1)

    print_json(data, title="Input JSON")

    # Validate
    from grammar_guard.validation import validate

    print_separator()
    print_info("Validating...")

    result = validate(json.dumps(data), schema)

    console.print()
    if result.is_valid:
        print_success("Validation passed!")
    else:
        print_error("Validation failed")
        print_validation_errors(result.errors)
        raise SystemExit(1)


def benchmark_command(
    schema_path: Path,
    prompts_path: Optional[Path],
    prompts_list: Optional[List[str]],
    model: str,
    backend: str,
    device: Optional[str],
    max_tokens: int,
    max_retries: int,
    iterations: int,
    show_outputs: bool
) -> None:
    """
    Execute the benchmark command.

    Args:
        schema_path: Path to JSON schema file
        prompts_path: Optional path to prompts file
        prompts_list: Optional list of prompts
        model: Model ID or path
        backend: Backend to use
        device: Device to use
        max_tokens: Maximum tokens to generate
        max_retries: Maximum retry attempts
        iterations: Number of iterations per prompt
        show_outputs: Whether to show generated outputs
    """
    print_header("GrammarGuard - Benchmark")

    # Load schema
    try:
        schema = load_schema_file(schema_path)
        print_success(f"Loaded schema from: {schema_path}")
    except Exception as e:
        print_error(f"Failed to load schema: {e}")
        raise SystemExit(1)

    # Load prompts
    if prompts_path:
        try:
            prompts = load_prompts_file(prompts_path)
            print_success(f"Loaded {len(prompts)} prompts from: {prompts_path}")
        except Exception as e:
            print_error(f"Failed to load prompts: {e}")
            raise SystemExit(1)
    elif prompts_list:
        prompts = prompts_list
        print_success(f"Using {len(prompts)} provided prompts")
    else:
        print_error("No prompts provided. Use --prompts-file or --prompts")
        raise SystemExit(1)

    # Configuration
    print_separator()
    print_info(f"Model: [bold]{model}[/bold]")
    print_info(f"Backend: [bold]{backend}[/bold]")
    print_info(f"Prompts: [bold]{len(prompts)}[/bold]")
    print_info(f"Iterations: [bold]{iterations}[/bold]")
    print_info(f"Total runs: [bold]{len(prompts) * iterations}[/bold]")
    print_separator()

    # Initialize generator
    print_model_loading(model, backend)

    try:
        from grammar_guard import GrammarConstrainedGenerator

        with create_progress_spinner() as progress:
            progress.add_task(description="Loading model...", total=None)

            generator = GrammarConstrainedGenerator(
                model=model,
                backend=backend,
                device=device
            )

        print_success("Model loaded successfully")

    except Exception as e:
        print_error(f"Failed to load model: {e}")
        raise SystemExit(1)

    # Run benchmark
    console.print()
    print_info("Running benchmark...")
    print_separator()

    results: List[Dict[str, Any]] = []

    for iteration in range(iterations):
        console.print(f"\n[bold cyan]Iteration {iteration + 1}/{iterations}[/bold cyan]\n")

        for i, prompt in enumerate(prompts, 1):
            console.print(f"[dim]Prompt {i}/{len(prompts)}:[/dim] {prompt[:60]}...")

            try:
                result = generator.generate(
                    prompt=prompt,
                    schema=schema,
                    max_tokens=max_tokens,
                    max_retries=max_retries
                )

                results.append({
                    "prompt": prompt,
                    "is_valid": result.is_valid,
                    "latency_ms": result.latency_ms,
                    "retries": result.retries,
                    "tokens_generated": result.tokens_generated,
                    "output": result.output
                })

                status = "[green]✓[/green]" if result.is_valid else "[red]✗[/red]"
                console.print(f"  {status} {result.latency_ms:.0f}ms, {result.retries} retries\n")

                if show_outputs and result.is_valid:
                    print_json(result.output, title=f"Output {i}")

            except Exception as e:
                print_warning(f"  Failed: {e}")
                results.append({
                    "prompt": prompt,
                    "is_valid": False,
                    "latency_ms": 0,
                    "retries": max_retries,
                    "tokens_generated": 0,
                    "output": ""
                })

    # Display results
    print_separator()
    print_benchmark_results(results)

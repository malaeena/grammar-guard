"""
Main CLI entry point using Typer.

This module defines the command-line interface for GrammarGuard using Typer.
It provides three main commands: generate, validate, and benchmark.
"""

from pathlib import Path
from typing import Optional, List

import typer
from typing_extensions import Annotated

from .commands import generate_command, validate_command, benchmark_command
from .display import print_error


# Create Typer app
app = typer.Typer(
    name="grammar-guard",
    help="GrammarGuard - Constrained JSON generation for LLMs",
    add_completion=False,
    rich_markup_mode="rich"
)


@app.command("generate")
def generate(
    prompt: Annotated[
        str,
        typer.Option("--prompt", "-p", help="Generation prompt")
    ],
    schema: Annotated[
        Path,
        typer.Option("--schema", "-s", help="Path to JSON schema file", exists=True, file_okay=True, dir_okay=False)
    ],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model ID (HuggingFace) or path (GGUF)")
    ] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    backend: Annotated[
        str,
        typer.Option("--backend", "-b", help="Backend to use: transformers or llamacpp")
    ] = "transformers",
    device: Annotated[
        Optional[str],
        typer.Option("--device", "-d", help="Device: cpu, cuda, mps, or None for auto-detect")
    ] = None,
    max_tokens: Annotated[
        int,
        typer.Option("--max-tokens", help="Maximum tokens to generate")
    ] = 200,
    max_retries: Annotated[
        int,
        typer.Option("--max-retries", help="Maximum retry attempts with schema simplification")
    ] = 3,
    temperature: Annotated[
        float,
        typer.Option("--temperature", "-t", help="Sampling temperature (0.0-2.0)")
    ] = 0.7,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Path to save output JSON")
    ] = None,
    show_schema: Annotated[
        bool,
        typer.Option("--show-schema", help="Display the schema before generation")
    ] = False,
    show_analysis: Annotated[
        bool,
        typer.Option("--show-analysis", help="Show detailed field analysis of output")
    ] = False,
) -> None:
    """
    Generate JSON conforming to a schema using constrained decoding.

    Example:
        grammar-guard generate \\
            --prompt "Generate a user profile for Alice, age 28" \\
            --schema schema.json \\
            --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \\
            --backend transformers \\
            --device mps \\
            --output result.json
    """
    try:
        generate_command(
            prompt=prompt,
            schema_path=schema,
            model=model,
            backend=backend,
            device=device,
            max_tokens=max_tokens,
            max_retries=max_retries,
            temperature=temperature,
            output_path=output,
            show_schema=show_schema,
            show_analysis=show_analysis
        )
    except Exception as e:
        print_error(f"Command failed: {e}")
        raise typer.Exit(code=1)


@app.command("validate")
def validate(
    json_file: Annotated[
        Path,
        typer.Option("--json", "-j", help="Path to JSON file to validate", exists=True, file_okay=True, dir_okay=False)
    ],
    schema: Annotated[
        Path,
        typer.Option("--schema", "-s", help="Path to JSON schema file", exists=True, file_okay=True, dir_okay=False)
    ],
    show_schema: Annotated[
        bool,
        typer.Option("--show-schema", help="Display the schema")
    ] = False,
) -> None:
    """
    Validate existing JSON against a schema.

    Example:
        grammar-guard validate \\
            --json output.json \\
            --schema schema.json \\
            --show-schema
    """
    try:
        validate_command(
            json_path=json_file,
            schema_path=schema,
            show_schema=show_schema
        )
    except Exception as e:
        print_error(f"Command failed: {e}")
        raise typer.Exit(code=1)


@app.command("benchmark")
def benchmark(
    schema: Annotated[
        Path,
        typer.Option("--schema", "-s", help="Path to JSON schema file", exists=True, file_okay=True, dir_okay=False)
    ],
    prompts_file: Annotated[
        Optional[Path],
        typer.Option("--prompts-file", help="Path to file with prompts (one per line)", exists=True, file_okay=True, dir_okay=False)
    ] = None,
    prompts: Annotated[
        Optional[List[str]],
        typer.Option("--prompt", "-p", help="Individual prompt (can be used multiple times)")
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model ID or path")
    ] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    backend: Annotated[
        str,
        typer.Option("--backend", "-b", help="Backend: transformers or llamacpp")
    ] = "transformers",
    device: Annotated[
        Optional[str],
        typer.Option("--device", "-d", help="Device: cpu, cuda, mps, or None for auto")
    ] = None,
    max_tokens: Annotated[
        int,
        typer.Option("--max-tokens", help="Maximum tokens to generate")
    ] = 200,
    max_retries: Annotated[
        int,
        typer.Option("--max-retries", help="Maximum retry attempts")
    ] = 2,
    iterations: Annotated[
        int,
        typer.Option("--iterations", "-i", help="Number of iterations per prompt")
    ] = 1,
    show_outputs: Annotated[
        bool,
        typer.Option("--show-outputs", help="Display generated outputs")
    ] = False,
) -> None:
    """
    Run benchmarks to measure validity rate and performance.

    Example:
        grammar-guard benchmark \\
            --schema schema.json \\
            --prompts-file prompts.txt \\
            --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \\
            --iterations 5
    """
    try:
        benchmark_command(
            schema_path=schema,
            prompts_path=prompts_file,
            prompts_list=prompts,
            model=model,
            backend=backend,
            device=device,
            max_tokens=max_tokens,
            max_retries=max_retries,
            iterations=iterations,
            show_outputs=show_outputs
        )
    except Exception as e:
        print_error(f"Command failed: {e}")
        raise typer.Exit(code=1)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", "-v", help="Show version and exit")
    ] = False,
) -> None:
    """
    GrammarGuard - Constrained JSON generation for LLMs.

    Ensures LLM outputs conform to JSON Schema through constrained decoding.
    """
    if version:
        from grammar_guard import __version__
        typer.echo(f"GrammarGuard version {__version__}")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def cli() -> None:
    """CLI entry point for poetry script."""
    app()


if __name__ == "__main__":
    cli()

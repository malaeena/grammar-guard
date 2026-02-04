"""
Rich terminal display utilities for CLI.

Provides formatted output using the Rich library for:
- Progress indicators
- Syntax-highlighted JSON
- Error messages with context
- Statistics tables
- Success/failure indicators
"""

import json
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text


console = Console()


def print_header(title: str) -> None:
    """Print a formatted header."""
    console.print()
    console.print(f"[bold cyan]{title}[/bold cyan]")
    console.print("=" * len(title))
    console.print()


def print_success(message: str) -> None:
    """Print a success message with checkmark."""
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    """Print an error message with X mark."""
    console.print(f"[red]✗[/red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow]⚠[/yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[blue]ℹ[/blue] {message}")


def print_json(data: Any, title: Optional[str] = None) -> None:
    """
    Print JSON data with syntax highlighting.

    Args:
        data: JSON-serializable data or JSON string
        title: Optional title for the panel
    """
    # Convert to JSON string if needed
    if isinstance(data, str):
        json_str = data
    else:
        json_str = json.dumps(data, indent=2)

    # Create syntax-highlighted JSON
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)

    if title:
        panel = Panel(syntax, title=f"[bold]{title}[/bold]", border_style="cyan")
        console.print(panel)
    else:
        console.print(syntax)


def print_schema(schema: Dict, title: str = "Schema") -> None:
    """Print a schema with syntax highlighting."""
    print_json(schema, title)


def print_validation_errors(errors: List[str]) -> None:
    """
    Print validation errors in a formatted list.

    Args:
        errors: List of validation error messages
    """
    if not errors:
        return

    console.print()
    console.print("[bold red]Validation Errors:[/bold red]")
    for error in errors:
        console.print(f"  [red]•[/red] {error}")
    console.print()


def print_result_stats(
    is_valid: bool,
    retries: int,
    latency_ms: float,
    tokens_generated: int,
    simplification_level: int = 0
) -> None:
    """
    Print generation result statistics in a table.

    Args:
        is_valid: Whether output is valid
        retries: Number of retries attempted
        latency_ms: Generation latency in milliseconds
        tokens_generated: Number of tokens generated
        simplification_level: Schema simplification level used
    """
    table = Table(title="Generation Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="white", width=30)

    # Valid status with color
    valid_text = Text("✓ Valid", style="green bold") if is_valid else Text("✗ Invalid", style="red bold")
    table.add_row("Status", valid_text)

    table.add_row("Retries", str(retries))
    table.add_row("Latency", f"{latency_ms:.0f} ms")
    table.add_row("Tokens Generated", str(tokens_generated))

    if simplification_level > 0:
        table.add_row("Simplification Level", f"{simplification_level}", style="yellow")

    console.print()
    console.print(table)
    console.print()


def print_benchmark_results(results: List[Dict]) -> None:
    """
    Print benchmark results in a formatted table.

    Args:
        results: List of result dictionaries with keys:
            - prompt: Test prompt
            - is_valid: Validation status
            - latency_ms: Generation latency
            - retries: Number of retries
            - tokens_generated: Token count
    """
    table = Table(title="Benchmark Results", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Prompt", style="cyan", width=40)
    table.add_column("Valid", justify="center", width=8)
    table.add_column("Latency (ms)", justify="right", width=12)
    table.add_column("Retries", justify="center", width=8)
    table.add_column("Tokens", justify="right", width=8)

    for i, result in enumerate(results, 1):
        prompt_preview = result["prompt"][:37] + "..." if len(result["prompt"]) > 40 else result["prompt"]
        valid_icon = "[green]✓[/green]" if result["is_valid"] else "[red]✗[/red]"

        table.add_row(
            str(i),
            prompt_preview,
            valid_icon,
            f"{result['latency_ms']:.0f}",
            str(result["retries"]),
            str(result.get("tokens_generated", "N/A"))
        )

    console.print()
    console.print(table)
    console.print()

    # Print summary statistics
    valid_count = sum(1 for r in results if r["is_valid"])
    total = len(results)
    avg_latency = sum(r["latency_ms"] for r in results) / total if total > 0 else 0
    avg_retries = sum(r["retries"] for r in results) / total if total > 0 else 0

    summary_table = Table(title="Summary", show_header=True, header_style="bold green")
    summary_table.add_column("Metric", style="cyan", width=25)
    summary_table.add_column("Value", style="white", width=25)

    validity_rate = (valid_count / total * 100) if total > 0 else 0
    summary_table.add_row("Validity Rate", f"{validity_rate:.1f}% ({valid_count}/{total})")
    summary_table.add_row("Average Latency", f"{avg_latency:.0f} ms")
    summary_table.add_row("Average Retries", f"{avg_retries:.2f}")

    console.print(summary_table)
    console.print()


def create_progress_spinner(message: str = "Generating...") -> Progress:
    """
    Create a progress spinner for long-running operations.

    Args:
        message: Message to display

    Returns:
        Progress context manager
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    )


def print_separator() -> None:
    """Print a visual separator line."""
    console.print("[dim]" + "─" * 70 + "[/dim]")


def print_field_analysis(data: Dict, title: str = "Field Analysis") -> None:
    """
    Print detailed field analysis for generated data.

    Args:
        data: Generated JSON data
        title: Title for the analysis section
    """
    console.print()
    console.print(f"[bold cyan]{title}:[/bold cyan]")

    def print_fields(obj: Any, indent: int = 0) -> None:
        """Recursively print fields with indentation."""
        prefix = "  " * indent

        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    console.print(f"{prefix}[cyan]{key}:[/cyan]")
                    print_fields(value, indent + 1)
                else:
                    value_str = str(value)
                    if isinstance(value, str):
                        value_str = f'"{value}" [dim](length: {len(value)})[/dim]'
                    console.print(f"{prefix}[cyan]{key}:[/cyan] {value_str}")

        elif isinstance(obj, list):
            console.print(f"{prefix}[dim]Array with {len(obj)} item(s)[/dim]")
            for i, item in enumerate(obj):
                console.print(f"{prefix}[yellow]Item {i + 1}:[/yellow]")
                print_fields(item, indent + 1)

    print_fields(data)
    console.print()


def confirm_action(message: str) -> bool:
    """
    Ask user for confirmation.

    Args:
        message: Confirmation message

    Returns:
        True if user confirmed, False otherwise
    """
    from rich.prompt import Confirm
    return Confirm.ask(message)


def print_model_loading(model_id: str, backend: str) -> None:
    """Print model loading information."""
    console.print()
    print_info(f"Loading model: [bold]{model_id}[/bold]")
    print_info(f"Backend: [bold]{backend}[/bold]")
    console.print()

"""
Unit tests for CLI module structure.

These tests verify the CLI module structure without requiring full dependencies.
"""

import pytest
from pathlib import Path


def test_cli_module_exists():
    """Test that CLI module exists."""
    cli_dir = Path(__file__).parent.parent.parent / "grammar_guard" / "cli"
    assert cli_dir.exists()


def test_cli_files_exist():
    """Test that all CLI files exist."""
    cli_dir = Path(__file__).parent.parent.parent / "grammar_guard" / "cli"

    expected_files = [
        "__init__.py",
        "main.py",
        "commands.py",
        "display.py"
    ]

    for filename in expected_files:
        filepath = cli_dir / filename
        assert filepath.exists(), f"Missing CLI file: {filename}"


def test_cli_main_structure():
    """Test that main.py has expected structure."""
    main_file = Path(__file__).parent.parent.parent / "grammar_guard" / "cli" / "main.py"
    content = main_file.read_text()

    # Check for key components
    assert "import typer" in content
    assert "def generate(" in content
    assert "def validate(" in content
    assert "def benchmark(" in content
    assert "def cli()" in content
    assert "app = typer.Typer(" in content


def test_cli_commands_structure():
    """Test that commands.py has expected structure."""
    commands_file = Path(__file__).parent.parent.parent / "grammar_guard" / "cli" / "commands.py"
    content = commands_file.read_text()

    # Check for key functions
    assert "def generate_command(" in content
    assert "def validate_command(" in content
    assert "def benchmark_command(" in content
    assert "def load_schema_file(" in content
    assert "def load_prompts_file(" in content


def test_cli_display_structure():
    """Test that display.py has expected structure."""
    display_file = Path(__file__).parent.parent.parent / "grammar_guard" / "cli" / "display.py"
    content = display_file.read_text()

    # Check for key functions
    assert "from rich.console import Console" in content
    assert "def print_json(" in content
    assert "def print_result_stats(" in content
    assert "def print_benchmark_results(" in content
    assert "def create_progress_spinner(" in content


def test_cli_init_exports():
    """Test that __init__.py exports app."""
    init_file = Path(__file__).parent.parent.parent / "grammar_guard" / "cli" / "__init__.py"
    content = init_file.read_text()

    assert "from .main import app" in content
    assert '__all__ = ["app"]' in content


def test_pyproject_has_cli_script():
    """Test that pyproject.toml has CLI script entry point."""
    pyproject_file = Path(__file__).parent.parent.parent / "pyproject.toml"
    content = pyproject_file.read_text()

    assert "[tool.poetry.scripts]" in content
    assert "grammar-guard = " in content
    assert "grammar_guard.cli.main:cli" in content


@pytest.mark.skip(reason="Requires typer and rich dependencies")
def test_cli_import():
    """Test that CLI can be imported (requires dependencies)."""
    try:
        from grammar_guard.cli import app
        assert app is not None
    except ImportError:
        pytest.skip("CLI dependencies not installed")


@pytest.mark.skip(reason="Requires full installation")
def test_cli_help():
    """Test CLI help command (requires full installation)."""
    import subprocess

    result = subprocess.run(
        ["grammar-guard", "--help"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert "GrammarGuard" in result.stdout
    assert "generate" in result.stdout
    assert "validate" in result.stdout
    assert "benchmark" in result.stdout

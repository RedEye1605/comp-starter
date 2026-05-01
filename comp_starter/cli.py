"""CLI entry point for comp-starter."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from comp_starter.generator import generate_project, submit_file

console = Console()

TEMPLATE_INFO = {
    "datathon": {
        "description": "Standard ML/datathon project with EDA, baseline model, and submission pipeline.",
        "details": "Includes: data/ dirs, notebooks/, src/ modules, configs/, submissions/",
    },
    "kaggle": {
        "description": "Kaggle competition setup — extends datathon with Kaggle API integration.",
        "details": "Adds: kaggle-specific metadata, auto data download via kaggle CLI",
    },
    "hackathon": {
        "description": "Hackathon setup — extends datathon with app/ structure for demos.",
        "details": "Adds: app/ directory with FastAPI skeleton, Dockerfile, streamlit dashboard",
    },
    "research": {
        "description": "Research paper project with LaTeX structure and experiment tracking.",
        "details": "Includes: paper/ LaTeX skeleton, experiments/, src/, configs/",
    },
}


@click.group()
@click.version_option(version="0.1.0", prog_name="comp-starter")
def main() -> None:
    """Comp Starter — One-command competition scaffold generator."""
    pass


@main.command()
@click.argument("name")
@click.option(
    "--type",
    "project_type",
    type=click.Choice(["datathon", "kaggle", "hackathon", "research"]),
    default="datathon",
    help="Project template type.",
)
@click.option("--kaggle", "kaggle_slug", default=None, help="Kaggle competition slug for data download.")
def init(name: str, project_type: str, kaggle_slug: str | None) -> None:
    """Generate a new competition project scaffold.

    NAME is the project directory name to create.
    """
    if project_type == "kaggle" and kaggle_slug is None:
        console.print("[yellow]Warning: Using 'kaggle' type without --kaggle SLUG. Data download will be skipped.[/yellow]")

    console.print(f"\n[bold cyan]🚀 Generating [white]{name}[/white] ({project_type} template)...[/bold cyan]\n")

    try:
        generate_project(name, project_type, kaggle_slug)
        console.print(f"[bold green]✅ Project '{name}' created successfully![/bold green]")
        console.print(f"\n   [dim]cd {name} && code .[/dim]\n")
    except FileExistsError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise SystemExit(1)


@main.command("templates")
def list_templates() -> None:
    """List available project templates."""
    table = Table(title="Available Templates", show_header=True, header_style="bold cyan")
    table.add_column("Template", style="bold")
    table.add_column("Description")
    table.add_column("Includes", style="dim")

    for name, info in TEMPLATE_INFO.items():
        table.add_row(name, info["description"], info["details"])

    console.print(table)


@main.command()
@click.option("--file", "filepath", required=True, type=click.Path(exists=True), help="Path to submission file.")
@click.option("--note", default="", help="Optional description for this submission version.")
def submit(filepath: str, note: str) -> None:
    """Auto-version a submission file.

    Copies the file to submissions/ with an auto-incremented version number.
    """
    try:
        dest = submit_file(filepath, note)
        console.print(f"[bold green]✅ Submission saved:[/bold green] {dest}")
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

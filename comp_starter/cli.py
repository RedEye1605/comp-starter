"""CLI entry point for comp-starter."""

from __future__ import annotations

import json

import click
from rich.console import Console
from rich.table import Table

from comp_starter.generator import generate_project, submit_file, list_custom_templates

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
    "custom": {
        "description": "Use a custom template from ~/Obsidian/RhendyVault/03_templates/.",
        "details": "Requires --custom-path pointing to a template directory",
    },
}

# Global flags
_json_output = False
_quiet_mode = False


def _out(msg: str = ""):
    if not _quiet_mode:
        console.print(msg)


@click.group()
@click.version_option(version="0.2.0", prog_name="comp-starter")
@click.option("--json", "json_flag", is_flag=True, help="Output as JSON")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-essential output")
def main(json_flag: bool, quiet: bool) -> None:
    """Comp Starter — One-command competition scaffold generator."""
    global _json_output, _quiet_mode
    _json_output = json_flag
    _quiet_mode = quiet


@main.command()
@click.argument("name")
@click.option(
    "--type",
    "project_type",
    type=click.Choice(["datathon", "kaggle", "hackathon", "research", "custom"]),
    default="datathon",
    help="Project template type.",
)
@click.option("--kaggle", "kaggle_slug", default=None, help="Kaggle competition slug for data download.")
@click.option("--custom-path", default=None, help="Path to custom template directory (for --type custom).")
@click.option("--no-exp-tracker", is_flag=True, help="Skip auto-init of exp-tracker.")
def init(name: str, project_type: str, kaggle_slug: str | None, custom_path: str | None, no_exp_tracker: bool) -> None:
    """Generate a new competition project scaffold.

    NAME is the project directory name to create.
    """
    if project_type == "custom" and not custom_path:
        raise click.ClickException("--custom-path is required when using --type custom")

    if project_type == "kaggle" and kaggle_slug is None:
        _out("[yellow]Warning: Using 'kaggle' type without --kaggle SLUG. Data download will be skipped.[/yellow]")

    _out(f"\n[bold cyan]🚀 Generating [white]{name}[/white] ({project_type} template)...[/bold cyan]\n")

    try:
        project_dir = generate_project(name, project_type, kaggle_slug, custom_path=custom_path)

        # Auto-init exp-tracker unless disabled
        if not no_exp_tracker:
            try:
                from exp_tracker.db import init_db as exp_init_db
                exp_init_db(project_dir)
                _out("[dim]  exp-tracker initialized in project[/dim]")
            except Exception:
                pass  # exp-tracker not installed, skip silently

        if _json_output:
            print(json.dumps({
                "status": "created",
                "name": name,
                "type": project_type,
                "path": str(project_dir),
                "exp_tracker": not no_exp_tracker,
            }))
        else:
            _out(f"[bold green]✅ Project '{name}' created successfully![/bold green]")
            _out(f"\n   [dim]cd {name} && code .[/dim]\n")
    except FileExistsError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise SystemExit(1)


@main.command("templates")
def list_tmpl() -> None:
    """List available project templates."""
    templates = list_custom_templates()

    if _json_output:
        print(json.dumps(templates))
        return

    table = Table(title="Available Templates", show_header=True, header_style="bold cyan")
    table.add_column("Template", style="bold")
    table.add_column("Description")

    for name, info in TEMPLATE_INFO.items():
        table.add_row(name, info["description"])

    if templates:
        table.add_row("─" * 20, "─" * 40)
        for t in templates:
            table.add_row(f"custom: {t['name']}", t.get("description", "Custom template"))

    console.print(table)


@main.command()
@click.option("--file", "filepath", required=True, type=click.Path(exists=True), help="Path to submission file.")
@click.option("--note", default="", help="Optional description for this submission version.")
def submit(filepath: str, note: str) -> None:
    """Auto-version a submission file."""
    try:
        dest = submit_file(filepath, note)
        if _json_output:
            print(json.dumps({"status": "submitted", "path": str(dest), "note": note}))
        else:
            _out(f"[bold green]✅ Submission saved:[/bold green] {dest}")
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

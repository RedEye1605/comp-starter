"""Project generation logic."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _get_template_dir(template_type: str) -> Path:
    """Resolve the base template directory, with inheritance."""
    # Inheritance chain: kaggle -> datathon, hackathon -> datathon
    if template_type in ("kaggle", "hackathon"):
        return TEMPLATES_DIR / "datathon"
    return TEMPLATES_DIR / template_type


def _render_templates(
    src_dir: Path,
    dest_dir: Path,
    variables: dict,
) -> list[Path]:
    """Render all Jinja2 templates from src_dir into dest_dir."""
    if not src_dir.exists():
        return []

    created: list[Path] = []

    # Build a list of all template files first
    template_files: list[tuple[Path, Path, str]] = []  # (src_file, rel_file, filename)
    for root, _dirs, files in os.walk(str(src_dir)):
        rel_root = Path(root).relative_to(src_dir)
        for filename in files:
            src_file = Path(root) / filename
            rel_file = rel_root / filename
            template_files.append((src_file, rel_file, filename))

    # Group by LaTeX vs non-LaTeX to use appropriate delimiters
    latex_exts = {".tex", ".bib", ".sty", ".cls"}

    for src_file, rel_file, filename in template_files:
        is_latex = any(
            src_file.name.endswith(ext + ".j2") or (not src_file.name.endswith(".j2") and src_file.suffix in latex_exts)
            for ext in latex_exts
        )

        if filename.endswith(".j2"):
            dest_rel = rel_file.parent / filename[:-3]
            env = Environment(
                loader=FileSystemLoader(str(src_dir)),
                keep_trailing_newline=True,
                block_start_string="<%" if is_latex else "{%",
                block_end_string="%>" if is_latex else "%}",
                variable_start_string="<<" if is_latex else "{{",
                variable_end_string=">>" if is_latex else "}}",
                comment_start_string="<#" if is_latex else "{#",
                comment_end_string="#>" if is_latex else "#}",
            )
            template = env.get_template(str(rel_file))
            rendered = template.render(**variables)
            dest_path = dest_dir / dest_rel
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_text(rendered)
        else:
            dest_path = dest_dir / rel_file
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dest_path)

        created.append(dest_path)

    return created


def _copy_static(src_dir: Path, dest_dir: Path) -> list[Path]:
    """Copy static (non-template) files."""
    if not src_dir.exists():
        return []

    created: list[Path] = []
    for root, _dirs, files in os.walk(str(src_dir)):
        rel_root = Path(root).relative_to(src_dir)

        for filename in files:
            if filename.endswith(".j2"):
                continue
            src_file = Path(root) / filename
            rel_file = rel_root / filename
            dest_path = dest_dir / rel_file
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dest_path)
            created.append(dest_path)

    return created


def _create_notebook(dest: Path, title: str, cells: list[dict]) -> None:
    """Create a valid .ipynb Jupyter notebook."""
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "cells": cells,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(nb, indent=1))


def _make_markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _cell_id(),
        "metadata": {},
        "source": [source],
    }


def _make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": _cell_id(),
        "metadata": {},
        "source": [source],
        "execution_count": None,
        "outputs": [],
    }


_cell_counter = 0


def _cell_id() -> str:
    global _cell_counter
    _cell_counter += 1
    import uuid

    return uuid.uuid4().hex[:8]


def _generate_datathon_notebooks(project_dir: Path, name: str) -> None:
    """Generate EDA and baseline notebooks for datathon/kaggle/hackathon."""
    # 01_eda.ipynb
    eda_cells = [
        _make_markdown_cell(f"# {name} — Exploratory Data Analysis\n\nUnderstand the data, distributions, missing values, and key patterns."),
        _make_code_cell(
            "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nplt.style.use('seaborn-v0_8-whitegrid')\nsns.set_palette('husl')\n%matplotlib inline"
        ),
        _make_markdown_cell("## Load Data"),
        _make_code_cell(
            "train = pd.read_csv('../data/raw/train.csv')\ntest = pd.read_csv('../data/raw/test.csv')\nprint(f'Train shape: {train.shape}')\nprint(f'Test shape: {test.shape}')\ntrain.head()"
        ),
        _make_markdown_cell("## Basic Info"),
        _make_code_cell("train.info()\ntrain.describe()"),
        _make_markdown_cell("## Missing Values"),
        _make_code_cell(
            "missing = train.isnull().sum()\nmissing_pct = (missing / len(train) * 100).round(2)\npd.DataFrame({'count': missing, 'pct': missing_pct}).sort_values('pct', ascending=False).head(20)"
        ),
        _make_markdown_cell("## Target Distribution"),
        _make_code_cell(
            "# TODO: Update column name\n# train['target'].hist(bins=50)\n# plt.title('Target Distribution')\n# plt.show()"
        ),
        _make_markdown_cell("## Correlation Analysis"),
        _make_code_cell(
            "# numeric_cols = train.select_dtypes(include=[np.number]).columns\n# fig, ax = plt.subplots(figsize=(12, 8))\n# sns.heatmap(train[numeric_cols].corr(), annot=True, fmt='.2f', ax=ax)\n# plt.title('Correlation Matrix')\n# plt.show()"
        ),
        _make_markdown_cell("## Key Takeaways\n\n- TODO: Document findings here\n- TODO: Note feature ideas\n- TODO: Note data quality issues"),
    ]
    _create_notebook(project_dir / "notebooks" / "01_eda.ipynb", f"{name} — EDA", eda_cells)

    # 02_baseline.ipynb
    baseline_cells = [
        _make_markdown_cell(f"# {name} — Baseline Model\n\nSimple baseline to establish a performance floor."),
        _make_code_cell(
            "import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import cross_val_score\nfrom sklearn.ensemble import RandomForestClassifier  # or Regressor\nimport sys\nsys.path.append('..')\nfrom src.data import load_data\nfrom src.features import build_features\nfrom src.models import train_model\nfrom src.evaluation import evaluate"
        ),
        _make_markdown_cell("## Load & Prepare Data"),
        _make_code_cell(
            "train, test = load_data()\nX_train, y_train = build_features(train, mode='train')\nX_test = build_features(test, mode='test')\nprint(f'Train: {X_train.shape}, Test: {X_test.shape}')"
        ),
        _make_markdown_cell("## Baseline Model"),
        _make_code_cell(
            "# Quick baseline with default parameters\nfrom sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)\nscores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')  # adjust metric\nprint(f'CV Score: {scores.mean():.4f} ± {scores.std():.4f}')"
        ),
        _make_markdown_cell("## Train Final & Predict"),
        _make_code_cell(
            "model.fit(X_train, y_train)\npreds = model.predict(X_test)\nprint(f'Predictions shape: {preds.shape}')"
        ),
        _make_markdown_cell("## Generate Submission"),
        _make_code_cell(
            "# from src.submit import create_submission\n# create_submission(preds, test, path='../submissions/baseline.csv')\n# print('Submission saved!')"
        ),
        _make_markdown_cell("## Next Steps\n\n- Feature engineering\n- Try LightGBM / XGBoost\n- Hyperparameter tuning\n- Ensemble methods"),
    ]
    _create_notebook(project_dir / "notebooks" / "02_baseline.ipynb", f"{name} — Baseline", baseline_cells)


def _generate_research_notebooks(project_dir: Path, name: str) -> None:
    """Generate experiment notebook for research template."""
    cells = [
        _make_markdown_cell(f"# {name} — Experiments\n\nTrack and reproduce experiments."),
        _make_code_cell(
            "import pandas as pd\nimport numpy as np\nimport json\nfrom pathlib import Path\n\nEXPERIMENTS_DIR = Path('../experiments/')\nEXPERIMENTS_DIR.mkdir(exist_ok=True)"
        ),
        _make_markdown_cell("## Experiment Log\n\n| ID | Description | Metric | Notes |\n|----|-------------|--------|-------|\n| 01 | Baseline    | -      | -     |"),
        _make_code_cell(
            "# Experiment tracking helper\ndef log_experiment(exp_id: str, description: str, metrics: dict, params: dict, notes: str = ''):\n    record = {\n        'id': exp_id,\n        'description': description,\n        'metrics': metrics,\n        'params': params,\n        'notes': notes,\n        'timestamp': pd.Timestamp.now().isoformat(),\n    }\n    path = EXPERIMENTS_DIR / f'{exp_id}.json'\n    with open(path, 'w') as f:\n        json.dump(record, f, indent=2)\n    print(f'Logged: {path}')\n    return record"
        ),
    ]
    _create_notebook(project_dir / "notebooks" / "01_experiments.ipynb", f"{name} — Experiments", cells)


def _init_git(project_dir: Path) -> None:
    """Initialize git repo if not already one."""
    if (project_dir / ".git").exists():
        return
    subprocess.run(["git", "init"], cwd=str(project_dir), capture_output=True, check=True)


def _try_kaggle_download(project_dir: Path, slug: str) -> None:
    """Attempt to download data via kaggle CLI."""
    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", slug, "-p", str(project_dir / "data" / "raw")],
            capture_output=True,
            check=True,
            timeout=120,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass  # kaggle CLI not available or competition not found — skip silently


def generate_project(name: str, project_type: str, kaggle_slug: str | None = None) -> Path:
    """Generate a full competition project scaffold."""
    project_dir = Path.cwd() / name

    if project_dir.exists():
        raise FileExistsError(f"Directory '{name}' already exists.")

    now = datetime.now()
    variables = {
        "name": name,
        "project_name": name.replace("-", " ").replace("_", " ").title(),
        "type": project_type,
        "kaggle_slug": kaggle_slug or "",
        "date": now.strftime("%Y-%m-%d"),
        "year": str(now.year),
    }

    # Create base directory
    project_dir.mkdir(parents=True)

    # --- Datathon base (used by datathon, kaggle, hackathon) ---
    if project_type in ("datathon", "kaggle", "hackathon"):
        base_dir = TEMPLATES_DIR / "datathon"
        _render_templates(base_dir, project_dir, variables)

        # Create data directories
        for d in ["data/raw", "data/processed", "data/external", "submissions", "models"]:
            (project_dir / d).mkdir(parents=True, exist_ok=True)

        # Generate notebooks
        _generate_datathon_notebooks(project_dir, name)

        # Kaggle-specific additions
        if project_type in ("kaggle",) or kaggle_slug:
            kaggle_dir = TEMPLATES_DIR / "kaggle"
            _render_templates(kaggle_dir, project_dir, variables)

        # Hackathon-specific additions
        if project_type == "hackathon":
            hack_dir = TEMPLATES_DIR / "hackathon"
            _render_templates(hack_dir, project_dir, variables)

    elif project_type == "research":
        base_dir = TEMPLATES_DIR / "research"
        _render_templates(base_dir, project_dir, variables)

        for d in ["experiments", "data", "figures"]:
            (project_dir / d).mkdir(parents=True, exist_ok=True)

        _generate_research_notebooks(project_dir, name)

    # Init git
    _init_git(project_dir)

    # Kaggle data download
    if kaggle_slug and project_type in ("kaggle", "datathon", "hackathon"):
        _try_kaggle_download(project_dir, kaggle_slug)

    return project_dir


def submit_file(filepath: str, note: str = "") -> Path:
    """Version and copy a submission file."""
    src = Path(filepath).resolve()
    if not src.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Find project root (look for submissions/ dir or use cwd)
    cwd = Path.cwd()
    submissions_dir = cwd / "submissions"
    if not submissions_dir.exists():
        submissions_dir.mkdir(parents=True)

    # Determine version number
    existing = sorted(submissions_dir.glob(f"{src.stem}_v*{src.suffix}"))
    if existing:
        last = existing[-1].stem
        try:
            last_ver = int(last.split("_v")[-1])
        except (ValueError, IndexError):
            last_ver = 0
    else:
        last_ver = 0

    next_ver = last_ver + 1
    dest_name = f"{src.stem}_v{next_ver:03d}{src.suffix}"
    dest = submissions_dir / dest_name

    shutil.copy2(src, dest)

    # Write metadata alongside
    meta = {
        "version": next_ver,
        "source": str(src),
        "note": note,
        "timestamp": datetime.now().isoformat(),
    }
    meta_path = submissions_dir / f"{src.stem}_v{next_ver:03d}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    return dest

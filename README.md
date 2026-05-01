# Comp Starter

One-command competition scaffold generator. Sets up the standard ML/Datathon project structure in seconds.

Every competition has the same setup: data dirs, EDA notebook, baseline model, submission pipeline, git init, requirements. This tool does it all in one command.

## Usage

```bash
# Create a new competition project
comp-starter init my-competition --type datathon

# With Kaggle integration
comp-starter init my-competition --kaggle SLUG --type kaggle

# List templates
comp-starter templates
```

## Generated Structure

```
my-competition/
├── README.md              # Competition info, links, deadlines
├── data/
│   ├── raw/               # Original data (gitignored)
│   ├── processed/         # Cleaned/feature-engineered data
│   └── external/          # External datasets
├── notebooks/
│   ├── 01_eda.ipynb       # Exploratory data analysis
│   ├── 02_baseline.ipynb  # First baseline model
│   └── 03_experiments.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py            # Data loading & preprocessing
│   ├── features.py        # Feature engineering
│   ├── models/            # Model definitions
│   ├── evaluation.py      # Metrics & validation
│   └── submit.py          # Submission generation
├── submissions/           # Versioned submission files
├── configs/
│   └── config.yaml        # Hyperparameters, paths
├── requirements.txt
└── .gitignore
```

## Features

- **Multiple templates:** datathon, kaggle, hackathon, research
- **Kaggle API integration:** Auto-download competition data
- **Git init:** With proper .gitignore for data/models
- **Customizable:** Add/remove sections via config
- **Submission versioning:** Auto-naming with timestamps

## License

MIT

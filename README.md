# Credit Card Approval Project

A professional machine learning system for automated credit card approval prediction using logistic regression, similar to what commercial banks use in production.

## Overview

Commercial banks receive a lot of applications for credit cards. Many of them get rejected for many reasons, like high loan balances, low income levels, or too many inquiries on an individual's credit report, for example. Manually analyzing these applications is mundane, error-prone, and time-consuming (and time is money!).

Luckily, this task can be automated with the power of machine learning and pretty much every commercial bank does so nowadays. In this project, we will build an automatic credit card approval predictor using Logistic regression machine learning technique, just like the real banks do.

## Features

- **Data Preprocessing**: Automated handling of missing values, encoding categorical variables, and data scaling
- **Model Training**: Logistic regression with hyperparameter tuning using GridSearchCV
- **Model Evaluation**: Comprehensive scoring and confusion matrix analysis
- **Production-Ready**: Structured codebase with proper separation of concerns
- **Quality Assurance**: Pre-commit hooks, linting, and testing infrastructure

## Project Structure

```
CreditCardApproval/
├── data/
│   └── cc_approvals.data              # Credit card approval dataset
├── src/
│   ├── data_preprocessing/
│   │   ├── __init__.py
│   │   ├── dataframe_manipulation.py  # Data loading and cleaning
│   │   └── dataframe_preprocessing.py  # Encoding and scaling
│   └── models/
│       ├── __init__.py
│       ├── preprocessing.py            # Model preprocessing utilities
│       ├── training.py                 # Model training functions
│       └── scoring.py                  # Model evaluation functions
├── scripts/
│   ├── __init__.py
│   └── train.py                        # Main training script
├── notebooks/                           # Jupyter notebooks for exploration
│   ├── Credit cards project.ipynb
│   └── ws.ipynb
├── config/
│   ├── __init__.py
│   └── config.yaml                     # Configuration file
├── tests/                               # Test directory
│   └── __init__.py
├── models/                              # Saved model files (gitignored)
│   └── .gitkeep
├── logs/                                # Log files (gitignored)
│   └── .gitkeep
├── utils/                               # Utility functions
│   └── __init__.py
├── pyproject.toml                       # Project configuration and dependencies
├── .pre-commit-config.yaml             # Pre-commit hooks configuration
└── README.md                            # This file
```

## Requirements

- Python >= 3.10
- Poetry (for dependency management)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd CreditCardApproval
   ```

2. **Install Poetry** (if not already installed):
   ```bash
   # Windows (PowerShell)
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

   # macOS/Linux
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install dependencies:**
   ```bash
   poetry install
   ```

4. **Activate the virtual environment:**
   ```bash
   poetry shell
   ```

## Usage

### Quick Start

Run the complete training pipeline:

```bash
poetry run python scripts/train.py
```

This will:
1. Load and preprocess the data
2. Split into train/test sets
3. Scale features
4. Train a logistic regression model
5. Perform hyperparameter tuning
6. Display results

### Programmatic Usage

You can also use the modules programmatically:

1. **Load and preprocess the data:**
   ```python
   from src.data_preprocessing import (
       load_data,
       handle_nan_values,
       rename_columns,
       encoding_the_columns,
       split_data,
       scale_data,
   )

   # Load data
   df = load_data()

   # Handle missing values
   df = handle_nan_values(df)

   # Rename columns
   df = rename_columns(df)

   # Encode categorical variables
   df = encoding_the_columns(df)

   # Split data
   X, y, X_train, X_test, y_train, y_test = split_data(df)

   # Scale features
   X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
   ```

2. **Train the model:**
   ```python
   from src.models import logistic_regression, best_logistic_regression

   # Train with default parameters
   accuracy, cm = logistic_regression(X_train_scaled, y_train)

   # Find best hyperparameters
   best_score, best_params = best_logistic_regression(X_train, y_train)
   ```

## Development

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality. Install them with:

```bash
poetry run pre-commit install
```

The hooks will automatically run on each commit and check for:
- Code formatting (Black)
- Linting (Pylint)
- Import sorting (isort)

### Running Tests

```bash
poetry run pytest
```

### Code Quality Tools

The project is configured with:
- **Black**: Code formatting
- **Pylint**: Code linting
- **isort**: Import sorting
- **mypy**: Type checking
- **flake8**: Additional linting

Run them individually:
```bash
poetry run black src/
poetry run pylint src/
poetry run isort src/
```

## Dependencies

### Core Dependencies
- `pandas >= 2.0.0` - Data manipulation
- `numpy < 2.0.0` - Numerical computing
- `scikit-learn < 1.6.0` - Machine learning algorithms
- `xgboost ^2.1.3` - Gradient boosting
- `mlflow == 2.15.1` - ML experiment tracking
- `optuna ^4.3.0` - Hyperparameter optimization
- `shap ^0.48.0` - Model interpretability

### Development Dependencies
- `black ^24.3.0` - Code formatter
- `pylint ^3.1.0` - Linter
- `pytest ^8.4.2` - Testing framework
- `pre-commit 3.5.0` - Git hooks

See `pyproject.toml` for the complete list of dependencies.

## License

This project is licensed under the MIT License.

## Author

**Ioannis**
- Email: jkioutsioukis@gmail.com

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

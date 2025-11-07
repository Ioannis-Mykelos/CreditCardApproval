"""
Main training script for credit card approval model.

This script orchestrates the entire ML pipeline:
1. Load and preprocess data
2. Train the model
3. Evaluate and save the model
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Imports must come after sys.path modification to ensure correct module resolution
from src.data_preprocessing.dataframe_manipulation import (
    load_data,
    handle_nan_values,
    rename_columns,
)
from src.data_preprocessing.dataframe_preprocessing import (
    encoding_the_columns,
    split_data,
    scale_data,
)
from src.models.preprocessing import logistic_regression, best_logistic_regression


def main():
    """Main training pipeline."""
    # pylint: disable=invalid-name
    # ML variable names (X, y, X_train, etc.) are standard conventions
    print("Starting Credit Card Approval Model Training Pipeline...")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/6] Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} records")

    # Step 2: Handle missing values
    print("\n[2/6] Handling missing values...")
    df = handle_nan_values(df)

    # Step 3: Rename columns
    print("\n[3/6] Renaming columns...")
    df = rename_columns(df)

    # Step 4: Encode categorical variables
    print("\n[4/6] Encoding categorical variables...")
    df = encoding_the_columns(df)

    # Step 5: Split data
    print("\n[5/6] Splitting data into train/test sets...")
    # X, y, X_train, X_test are standard ML variable names (features and target)
    _, _, X_train, X_test, y_train, _ = split_data(df)
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Step 6: Scale features
    print("\n[6/6] Scaling features...")
    # X_test_scaled is unused but kept for potential future evaluation
    X_train_scaled, _ = scale_data(X_train, X_test)

    # Step 7: Train model with default parameters
    print("\n" + "=" * 60)
    print("Training Logistic Regression Model...")
    print("=" * 60)
    accuracy, confusion_matrix = logistic_regression(X_train_scaled, y_train)
    print(f"\nModel Accuracy: {accuracy:.4f}")

    # Step 8: Hyperparameter tuning
    print("\n" + "=" * 60)
    print("Hyperparameter Tuning...")
    print("=" * 60)
    best_score, best_params = best_logistic_regression(X_train, y_train)
    print(f"\nBest CV Score: {best_score:.4f}")
    print(f"Best Parameters: {best_params}")

    print("\n" + "=" * 60)
    print("Training pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

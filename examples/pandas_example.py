"""Example: Pandas data cleaning pipeline."""

import pandas as pd
import numpy as np
from data_cleaning_pipeline import DataCleaningPipeline, auto_clean_dataframe, data_quality_checks


def create_sample_data():
    """Create a sample messy dataset."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "id": range(1, n + 1),
        "name": ["Alice", "Bob", "Charlie", None, "  Alice  "] * (n // 5 + 1),
        "age": [25, 30, 35, 40, 45] * (n // 5 + 1),
        "salary": [50000, 60000, 70000, 80000, 90000] * (n // 5 + 1),
        "department": ["Engineering", "Sales", "HR", "Engineering", "Sales"] * (n // 5 + 1),
        "email": [
            "alice@example.com",
            "bob@example.com",
            "charlie@example.com",
            None,
            "  ALICE@EXAMPLE.COM  "
        ] * (n // 5 + 1),
    })


def main():
    df = create_sample_data()
    print(f"Original shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")

    # Method 1: Using the DataCleaningPipeline class
    pipeline = DataCleaningPipeline(df)
    cleaned_df = (
        pipeline
        .remove_duplicates()
        .handle_missing_values(strategy="auto")
        .normalize_text()
        .validate_ranges({"age": (0, 120), "salary": (0, 200000)})
        .get_cleaned_data()
    )

    print("\n" + pipeline.generate_report())

    # Method 2: Using the auto_clean function
    cleaned_df2 = auto_clean_dataframe(df)

    # Run quality checks
    data_quality_checks(
        cleaned_df,
        expected_types={"id": "int64", "age": "int64"},
        required_columns=["id", "name", "age"],
        range_checks={"age": (0, 120)},
        id_column="id",
        email_column="email",
    )


if __name__ == "__main__":
    main()

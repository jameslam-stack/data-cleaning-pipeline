"""Auto-cleaning function for quick data cleaning."""

import pandas as pd
import numpy as np
from data_cleaning_pipeline.pandas_pipeline import DataCleaningPipeline
from typing import Optional, Dict, Any


def auto_clean_dataframe(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Automatically clean a DataFrame with sensible defaults.

    This function provides a quick one-call interface for common data cleaning
    operations. It applies all cleaning steps with sensible defaults and can
    be customized via the config parameter.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to clean.
    config : dict, optional
        Dictionary configuring which steps to apply.
        Default steps applied if config is None:
        - 'remove_duplicates': True
        - 'handle_missing': True
        - 'remove_outliers': True
        - 'fix_types': True
        - 'normalize_text': True
        - 'drop_empty_columns': True
        - 'drop_empty_rows': True

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame.

    Examples
    --------
    >>> df = pd.read_csv('messy_data.csv')
    >>> cleaned = auto_clean_dataframe(df)
    >>>
    >>> # With custom config
    >>> cleaned = auto_clean_dataframe(df, config={
    ...     'remove_duplicates': True,
    ...     'handle_missing': False,
    ...     'normalize_text': True,
    ... })
    """
    if config is None:
        config = {
            "remove_duplicates": True,
            "handle_missing": True,
            "remove_outliers": True,
            "fix_types": True,
            "normalize_text": True,
            "drop_empty_columns": True,
            "drop_empty_rows": True,
            "validate_ranges": False,
        }

    pipeline = DataCleaningPipeline(df)

    original_shape = df.shape
    print(f"Original shape: {original_shape}")

    # Drop empty columns
    if config.get("drop_empty_columns", False):
        empty_cols = df.columns[df.isnull().all() | (df == "").all()]
        if len(empty_cols) > 0:
            pipeline = pipeline.drop_columns(empty_cols.tolist())
            print(f"Dropped empty columns: {list(empty_cols)}")

    # Drop rows that are entirely empty
    if config.get("drop_empty_rows", False):
        before = len(pipeline.df)
        pipeline.df = pipeline.df.dropna(how="all")
        after = len(pipeline.df)
        if before != after:
            print(f"Dropped {before - after} fully empty rows")

    # Remove duplicates
    if config.get("remove_duplicates", False):
        pipeline = pipeline.remove_duplicates()
        print(f"After removing duplicates: {pipeline.df.shape}")

    # Handle missing values
    if config.get("handle_missing", False):
        numeric_cols = pipeline.df.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        for col in numeric_cols:
            if pipeline.df[col].isnull().sum() > 0:
                median_val = pipeline.df[col].median()
                pipeline.df[col] = pipeline.df[col].fillna(median_val)
                print(f"Filled '{col}' with median: {median_val:.2f}")

        categorical_cols = pipeline.df.select_dtypes(
            include=["object"]
        ).columns.tolist()
        for col in categorical_cols:
            if pipeline.df[col].isnull().sum() > 0:
                if pipeline.df[col].mode().empty:
                    pipeline.df[col] = pipeline.df[col].fillna("Unknown")
                    print(f"Filled '{col}' with 'Unknown'")
                else:
                    mode_val = pipeline.df[col].mode()[0]
                    pipeline.df[col] = pipeline.df[col].fillna(mode_val)
                    print(f"Filled '{col}' with mode: {mode_val}")

        print(f"After handling missing values: {pipeline.df.shape}")

    # Fix data types
    if config.get("fix_types", False):
        pipeline = pipeline.fix_data_types()
        print(f"After fixing types: {pipeline.df.shape}")

    # Normalize text
    if config.get("normalize_text", False):
        pipeline = pipeline.normalize_text()
        print(f"After normalizing text: {pipeline.df.shape}")

    # Remove outliers
    if config.get("remove_outliers", False):
        numeric_cols = pipeline.df.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        if numeric_cols:
            before = len(pipeline.df)
            pipeline = pipeline.remove_outliers(
                columns=numeric_cols,
                method="iqr",
                threshold=1.5,
            )
            after = len(pipeline.df)
            print(f"Removed {before - after} outlier rows")
            print(f"After removing outliers: {pipeline.df.shape}")

    # Validate ranges
    if config.get("validate_ranges", False) and isinstance(config.get("range_checks"), dict):
        pipeline = pipeline.validate_ranges(config["range_checks"])

    cleaned_df = pipeline.get_cleaned_data()
    print(f"Final cleaned shape: {cleaned_df.shape}")
    return cleaned_df
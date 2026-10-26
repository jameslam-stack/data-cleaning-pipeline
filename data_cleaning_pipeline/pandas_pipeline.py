"""Pandas data cleaning pipeline implementation."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Callable, Any, Union


class DataCleaningPipeline:
    """Data cleaning pipeline for pandas DataFrames.

    Provides a fluent interface for common data cleaning operations including
    removing duplicates, handling missing values, fixing data types, removing
    outliers, and normalizing text.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to clean.

    Examples
    --------
    >>> df = pd.read_csv('data.csv')
    >>> pipeline = DataCleaningPipeline(df)
    >>> cleaned_df = (pipeline
    ...     .remove_duplicates()
    ...     .handle_missing_values()
    ...     .remove_outliers()
    ...     .get_cleaned_data()
    ... )
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        self.cleaning_log: List[str] = []

    def _log(self, message: str) -> "DataCleaningPipeline":
        """Add a log entry and print the message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.cleaning_log.append(log_entry)
        print(message)
        return self

    def remove_duplicates(
        self,
        subset: Optional[List[str]] = None,
        keep: str = "first"
    ) -> "DataCleaningPipeline":
        """Remove duplicate rows from the DataFrame.

        Parameters
        ----------
        subset : list of str, optional
            Columns to consider for identifying duplicates.
            If None, all columns are used.
        keep : str, default 'first'
            Which duplicate entry to keep.
            - 'first': Drop duplicates except for the first occurrence.
            - 'last': Drop duplicates except for the last occurrence.
            - False: Drop all duplicates.

        Returns
        -------
        Self
            Returns the pipeline object for method chaining.
        """
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed = before - len(self.df)
        self._log(f"Removed {removed} duplicate rows")
        return self

    def handle_missing_values(
        self,
        strategy: str = "auto",
        drop_threshold: float = 50.0,
        numeric_fill: Optional[str] = None,
        categorical_fill: Optional[str] = None,
    ) -> "DataCleaningPipeline":
        """Handle missing values in the DataFrame.

        Parameters
        ----------
        strategy : str, default 'auto'
            Strategy for filling missing values:
            - 'auto': Automatically choose based on column type
            - 'mean': Fill numeric columns with mean
            - 'median': Fill numeric columns with median
            - 'mode': Fill categorical columns with mode
        drop_threshold : float, default 50.0
            Drop columns with more than this percentage of missing values.
        numeric_fill : str, optional
            Override numeric fill strategy ('mean', 'median').
        categorical_fill : str, optional
            Override categorical fill strategy ('mode', 'unknown').

        Returns
        -------
        Self
            Returns the pipeline object for method chaining.
        """
        missing = self.df.isnull().sum()
        columns_with_missing = missing[missing > 0]

        for col in columns_with_missing.index:
            missing_pct = (missing[col] / len(self.df)) * 100

            if missing_pct > drop_threshold:
                self._log(
                    f"Dropping column '{col}' ({missing_pct:.1f}% missing)"
                )
                self.df = self.df.drop(columns=[col])
                continue

            if self.df[col].dtype in ["int64", "int32", "float64", "float32"]:
                fill_strategy = numeric_fill if numeric_fill else strategy
                if fill_strategy == "mean":
                    fill_value = self.df[col].mean()
                elif fill_strategy == "median":
                    fill_value = self.df[col].median()
                else:  # auto or unspecified
                    fill_value = self.df[col].median() if strategy != "mean" else self.df[col].mean()

                self.df[col] = self.df[col].fillna(fill_value)
                self._log(f"Filled '{col}' with {fill_strategy}: {fill_value:.2f}")
            else:
                fill_strategy = categorical_fill if categorical_fill else strategy
                if fill_strategy == "mode":
                    mode_vals = self.df[col].mode()
                    fill_value = mode_vals[0] if len(mode_vals) > 0 else "Unknown"
                else:  # auto or unspecified
                    fill_value = "Unknown"
                self.df[col] = self.df[col].fillna(fill_value)
                self._log(f"Filled '{col}' with: {fill_value}")

        return self

    def fix_data_types(
        self,
        type_mapping: Optional[Dict[str, type]] = None,
        auto_date_detection: bool = True,
    ) -> "DataCleaningPipeline":
        """Convert columns to appropriate data types.

        Parameters
        ----------
        type_mapping : dict, optional
            Mapping of column names to target dtypes.
            Example: {'age': int, 'date': 'datetime64[ns]'}
        auto_date_detection : bool, default True
            Automatically detect and convert columns with 'date' or 'time'
            in their names to datetime.

        Returns
        -------
        Self
            Returns the pipeline object for method chaining.
        """
        if type_mapping is None:
            type_mapping = {}

        for col in self.df.columns:
            if col in type_mapping:
                try:
                    self.df[col] = self.df[col].astype(type_mapping[col])
                    self._log(f"Converted '{col}' to {type_mapping[col]}")
                except Exception as e:
                    self._log(f"Failed to convert '{col}': {e}")
            elif auto_date_detection:
                if "date" in col.lower() or "time" in col.lower():
                    try:
                        self.df[col] = pd.to_datetime(self.df[col])
                        self._log(f"Converted '{col}' to datetime")
                    except Exception:
                        pass

        return self

    def remove_outliers(
        self,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> "DataCleaningPipeline":
        """Remove outliers using IQR or Z-score method.

        Parameters
        ----------
        columns : list, optional
            List of numeric columns to check for outliers.
            If None, all numeric columns are used.
        method : str, default 'iqr'
            Method for outlier detection:
            - 'iqr': Interquartile range method
            - 'zscore': Z-score method
        threshold : float, default 1.5
            Threshold for outlier detection.
            - For IQR: values outside Q1 - threshold*IQR to Q3 + threshold*IQR
            - For Z-score: absolute z-score above threshold

        Returns
        -------
        Self
            Returns the pipeline object for method chaining.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if len(columns) == 0:
            return self

        before = len(self.df)
        mask = pd.Series([True] * len(self.df))

        for col in columns:
            if col not in self.df.columns:
                continue

            if method == "iqr":
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR == 0:
                    continue
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                col_mask = (self.df[col] >= lower) & (self.df[col] <= upper)
            else:  # zscore
                if self.df[col].std() == 0:
                    continue
                z_scores = np.abs(
                    (self.df[col] - self.df[col].mean()) / self.df[col].std()
                )
                col_mask = z_scores < threshold

            mask = mask & col_mask

        self.df = self.df[mask]
        removed = before - len(self.df)
        self._log(f"Removed {removed} outlier rows using {method} method")
        return self

    def normalize_text(
        self,
        columns: Optional[List[str]] = None,
        lowercase: bool = True,
        strip_whitespace: bool = True,
        remove_special_chars: bool = False,
    ) -> "DataCleaningPipeline":
        """Normalize text columns.

        Parameters
        ----------
        columns : list, optional
            List of columns to normalize. If None, all object columns are used.
        lowercase : bool, default True
            Convert text to lowercase.
        strip_whitespace : bool, default True
            Strip leading and trailing whitespace.
        remove_special_chars : bool, default False
            Remove special characters (keeps alphanumeric and spaces).

        Returns
        -------
        Self
            Returns the pipeline object for method chaining.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=["object"]).columns.tolist()

        for col in columns:
            if col not in self.df.columns:
                continue

            if strip_whitespace:
                self.df[col] = self.df[col].str.strip()
            if lowercase:
                self.df[col] = self.df[col].str.lower()
            if remove_special_chars:
                import re
                self.df[col] = self.df[col].str.replace(r"[^\w\s]", "", regex=True)
            self._log(f"Normalized text in '{col}'")

        return self

    def encode_categorical(
        self,
        columns: Optional[List[str]] = None,
        method: str = "label",
    ) -> "DataCleaningPipeline":
        """Encode categorical variables.

        Parameters
        ----------
        columns : list, optional
            List of columns to encode. If None, all object columns are used.
        method : str, default 'label'
            Encoding method:
            - 'label': Label encoding (converts to integer codes)
            - 'onehot': One-hot encoding (creates new columns)

        Returns
        -------
        Self
            Returns the pipeline object for method chaining.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=["object"]).columns.tolist()

        for col in columns:
            if col not in self.df.columns:
                continue

            if method == "label":
                self.df[col] = pd.Categorical(self.df[col]).codes
                self._log(f"Label encoded '{col}'")
            elif method == "onehot":
                dummies = pd.get_dummies(self.df[col], prefix=col)
                self.df = pd.concat([self.df.drop(columns=[col]), dummies], axis=1)
                self._log(f"One-hot encoded '{col}'")

        return self

    def validate_ranges(
        self,
        range_checks: Dict[str, tuple],
        remove_invalid: bool = True,
    ) -> "DataCleaningPipeline":
        """Validate numeric columns are within expected ranges.

        Parameters
        ----------
        range_checks : dict
            Dictionary mapping column names to (min, max) tuples.
            Example: {'age': (0, 120), 'price': (0, 1000000)}
        remove_invalid : bool, default True
            If True, remove rows with values outside the expected range.
            If False, just log warnings.

        Returns
        -------
        Self
            Returns the pipeline object for method chaining.
        """
        for col, (min_val, max_val) in range_checks.items():
            if col not in self.df.columns:
                continue

            invalid = ((self.df[col] < min_val) | (self.df[col] > max_val)).sum()
            if invalid > 0:
                self._log(
                    f"WARNING: {invalid} values in '{col}' outside range "
                    f"[{min_val}, {max_val}]"
                )
                if remove_invalid:
                    self.df = self.df[
                        (self.df[col] >= min_val) & (self.df[col] <= max_val)
                    ]

        return self

    def rename_columns(
        self,
        rename_map: Dict[str, str],
    ) -> "DataCleaningPipeline":
        """Rename columns.

        Parameters
        ----------
        rename_map : dict
            Dictionary mapping old column names to new ones.

        Returns
        -------
        Self
            Returns the pipeline object for method chaining.
        """
        self.df = self.df.rename(columns=rename_map)
        self._log(f"Renamed columns: {rename_map}")
        return self

    def drop_columns(
        self,
        columns: List[str],
    ) -> "DataCleaningPipeline":
        """Drop specified columns.

        Parameters
        ----------
        columns : list
            List of column names to drop.

        Returns
        -------
        Self
            Returns the pipeline object for method chaining.
        """
        existing_cols = [c for c in columns if c in self.df.columns]
        if existing_cols:
            self.df = self.df.drop(columns=existing_cols)
            self._log(f"Dropped columns: {existing_cols}")
        return self

    def generate_report(self) -> str:
        """Generate a cleaning report.

        Returns
        -------
        str
            A formatted report of all cleaning operations performed.
        """
        report = f"""
Data Cleaning Report
====================

Original Shape: {self.original_shape}
Final Shape: {self.df.shape}
Rows Removed: {self.original_shape[0] - self.df.shape[0]}
Columns Removed: {self.original_shape[1] - self.df.shape[1]}

Cleaning Steps:
"""
        for step in self.cleaning_log:
            report += f"  - {step}\n"

        return report

    def get_cleaned_data(self) -> pd.DataFrame:
        """Return the cleaned DataFrame.

        Returns
        -------
        pd.DataFrame
            The cleaned DataFrame.
        """
        return self.df.copy()

    def save_report(self, filepath: str) -> None:
        """Save the cleaning report to a file.

        Parameters
        ----------
        filepath : str
            Path to save the report.
        """
        with open(filepath, "w") as f:
            f.write(self.generate_report())

    def save_cleaned_data(
        self,
        filepath: str,
        index: bool = False,
        format: Optional[str] = None,
    ) -> None:
        """Save the cleaned data to a file.

        Parameters
        ----------
        filepath : str
            Path to save the data.
        index : bool, default False
            Whether to write row index.
        format : str, optional
            File format ('csv', 'json', 'excel', 'parquet').
            If None, inferred from filepath extension.
        """
        if format is None:
            format = filepath.split(".")[-1].lower()

        if format == "csv":
            self.df.to_csv(filepath, index=index)
        elif format == "json":
            self.df.to_json(filepath, orient="records", indent=2)
        elif format == "excel":
            self.df.to_excel(filepath, index=index)
        elif format == "parquet":
            self.df.to_parquet(filepath, index=index)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self._log(f"Saved cleaned data to {filepath}")
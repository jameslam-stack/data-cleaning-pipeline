"""Polars data cleaning pipeline implementation."""

import polars as pl
from typing import Optional, Dict, List, Any, Union


class PolarsPipeline:
    """Data cleaning pipeline for Polars DataFrames.

    Provides fast data cleaning operations using Polars' lazy evaluation.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame to clean.

    Examples
    --------
    >>> df = pl.read_csv('data.csv')
    >>> pipeline = PolarsPipeline(df)
    >>> cleaned_df = pipeline.clean()
    """

    def __init__(self, df: pl.DataFrame):
        self.df = df
        self.original_shape = df.shape
        self.steps: List[str] = []
        self._log(f"Initialized pipeline with shape {df.shape}")

    def _log(self, message: str) -> "PolarsPipeline":
        """Log a pipeline step."""
        self.steps.append(message)
        print(message)
        return self

    def remove_duplicates(
        self,
        subset: Optional[List[str]] = None,
        keep: str = "first",
    ) -> "PolarsPipeline":
        """Remove duplicate rows.

        Parameters
        ----------
        subset : list, optional
            Columns to consider for identifying duplicates.
        keep : str, default 'first'
            Which duplicate entry to keep ('first' or 'last').

        Returns
        -------
        Self
            Returns the pipeline object for method chaining.
        """
        before = self.df.height
        if subset:
            self.df = self.df.unique(subset=subset, keep=keep)
        else:
            self.df = self.df.unique(keep=keep)
        removed = before - self.df.height
        self._log(f"Removed {removed} duplicate rows")
        return self

    def handle_missing_values(
        self,
        strategy: str = "auto",
        numeric_fill: Optional[str] = None,
        categorical_fill: Optional[str] = None,
    ) -> "PolarsPipeline":
        """Handle missing values in the DataFrame.

        Parameters
        ----------
        strategy : str, default 'auto'
            Strategy for filling missing values:
            - 'auto': Automatically choose based on column type
            - 'mean': Fill numeric columns with mean
            - 'median': Fill numeric columns with median
        numeric_fill : str, optional
            Override numeric fill strategy ('mean', 'median').
        categorical_fill : str, optional
            Override categorical fill strategy ('mode', 'unknown').

        Returns
        -------
        Self
            Returns the pipeline object for method chaining.
        """
        numeric_cols = self.df.select(pl.col(
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64
        )).columns

        text_cols = self.df.select(pl.col(pl.Utf8)).columns

        for col in numeric_cols:
            fill_strategy = numeric_fill if numeric_fill else strategy
            if fill_strategy == "mean":
                value = self.df.select(pl.col(col).mean()).item()
            else:
                value = self.df.select(pl.col(col).median()).item()
            self.df = self.df.with_columns(pl.col(col).fill_null(value))
            self._log(f"Filled '{col}' with {fill_strategy}: {value:.2f}")

        for col in text_cols:
            fill_strategy = categorical_fill if categorical_fill else strategy
            if fill_strategy == "mode":
                mode_val = self.df.select(pl.col(col).mode()).item()
            else:
                mode_val = "Unknown"
            self.df = self.df.with_columns(pl.col(col).fill_null(mode_val))
            self._log(f"Filled '{col}' with: {mode_val}")

        return self

    def fix_data_types(
        self,
        type_mapping: Optional[Dict[str, str]] = None,
    ) -> "PolarsPipeline":
        """Convert columns to appropriate data types.

        Parameters
        ----------
        type_mapping : dict, optional
            Mapping of column names to Polars type strings.
            Example: {'age': 'Int64', 'date': 'Date'}

        Returns
        -------
        Self
            Returns the pipeline object for method chaining.
        """
        if type_mapping is None:
            type_mapping = {}

        for col, dtype in type_mapping.items():
            if col in self.df.columns:
                try:
                    self.df = self.df.with_columns(pl.col(col).cast(dtype))
                    self._log(f"Converted '{col}' to {dtype}")
                except Exception as e:
                    self._log(f"Failed to convert '{col}': {e}")

        return self

    def remove_outliers(
        self,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> "PolarsPipeline":
        """Remove outliers using IQR or Z-score method.

        Parameters
        ----------
        columns : list, optional
            List of numeric columns to check for outliers.
        method : str, default 'iqr'
            Method for outlier detection ('iqr' or 'zscore').
        threshold : float, default 1.5
            Threshold for outlier detection.

        Returns
        -------
        Self
            Returns the pipeline object for method chaining.
        """
        if columns is None:
            columns = self.df.select(pl.col(pl.Numeric)).columns

        for col in columns:
            if col not in self.df.columns:
                continue

            if method == "iqr":
                Q1 = self.df.select(pl.col(col).quantile(0.25)).item()
                Q3 = self.df.select(pl.col(col).quantile(0.75)).item()
                IQR = Q3 - Q1
                if IQR == 0:
                    continue
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                self.df = self.df.filter(
                    (pl.col(col) >= lower) & (pl.col(col) <= upper)
                )
            else:
                std = self.df.select(pl.col(col).std()).item()
                if std == 0:
                    continue
                mean = self.df.select(pl.col(col).mean()).item()
                self.df = self.df.filter(
                    (pl.col(col) - mean).abs() < threshold * std
                )

        self._log(f"Removed outliers using {method} method")
        return self

    def normalize_text(
        self,
        columns: Optional[List[str]] = None,
        lowercase: bool = True,
        strip_whitespace: bool = True,
    ) -> "PolarsPipeline":
        """Normalize text columns.

        Parameters
        ----------
        columns : list, optional
            List of columns to normalize.
        lowercase : bool, default True
            Convert text to lowercase.
        strip_whitespace : bool, default True
            Strip leading and trailing whitespace.

        Returns
        -------
        Self
            Returns the pipeline object for method chaining.
        """
        if columns is None:
            columns = self.df.select(pl.col(pl.Utf8)).columns

        for col in columns:
            if col not in self.df.columns:
                continue

            expr = pl.col(col)
            if strip_whitespace:
                expr = expr.str.strip()
            if lowercase:
                expr = expr.str.to_lowercase()
            self.df = self.df.with_columns(expr)
            self._log(f"Normalized text in '{col}'")

        return self

    def validate_ranges(
        self,
        range_checks: Dict[str, tuple],
    ) -> "PolarsPipeline":
        """Validate numeric columns are within expected ranges.

        Parameters
        ----------
        range_checks : dict
            Dictionary mapping column names to (min, max) tuples.

        Returns
        -------
        Self
            Returns the pipeline object for method chaining.
        """
        for col, (min_val, max_val) in range_checks.items():
            if col not in self.df.columns:
                continue

            invalid = self.df.filter(
                (pl.col(col) < min_val) | (pl.col(col) > max_val)
            ).height
            if invalid > 0:
                self._log(
                    f"WARNING: {invalid} values in '{col}' outside range "
                    f"[{min_val}, {max_val}]"
                )
                self.df = self.df.filter(
                    (pl.col(col) >= min_val) & (pl.col(col) <= max_val)
                )

        return self

    def encode_categorical(
        self,
        columns: Optional[List[str]] = None,
        method: str = "label",
    ) -> "PolarsPipeline":
        """Encode categorical variables.

        Parameters
        ----------
        columns : list, optional
            List of columns to encode.
        method : str, default 'label'
            Encoding method ('label' or 'onehot').

        Returns
        -------
        Self
            Returns the pipeline object for method chaining.
        """
        if columns is None:
            columns = self.df.select(pl.col(pl.Utf8)).columns

        for col in columns:
            if col not in self.df.columns:
                continue

            if method == "label":
                categories = self.df.select(pl.col(col).unique()).to_series()
                category_map = {v: i for i, v in enumerate(categories)}
                self.df = self.df.with_columns(
                    pl.col(col).replace(category_map).alias(col)
                )
                self._log(f"Label encoded '{col}'")
            elif method == "onehot":
                dummies = self.df.select(
                    pl.col(col).str.to_uppercase().alias(f"{col}_cat")
                ).to_dummies(columns=[f"{col}_cat"])
                self.df = self.df.with_columns(dummies)
                self._log(f"One-hot encoded '{col}'")

        return self

    def generate_report(self) -> str:
        """Generate a cleaning report.

        Returns
        -------
        str
            A formatted report of all cleaning operations performed.
        """
        report = f"""
Polars Data Cleaning Report
===========================

Original Shape: {self.original_shape}
Final Shape: {self.df.shape}
Rows Removed: {self.original_shape[0] - self.df.shape[0]}
Columns Removed: {self.original_shape[1] - self.df.shape[1]}

Cleaning Steps:
"""
        for step in self.steps:
            report += f"  - {step}\n"

        return report

    def clean(self) -> pl.DataFrame:
        """Run the default cleaning pipeline.

        Returns
        -------
        pl.DataFrame
            The cleaned DataFrame.
        """
        return (self
            .remove_duplicates()
            .handle_missing_values()
            .normalize_text()
            .validate_ranges({})
            .get_cleaned_data())

    def get_cleaned_data(self) -> pl.DataFrame:
        """Return the cleaned DataFrame.

        Returns
        -------
        pl.DataFrame
            The cleaned DataFrame.
        """
        return self.df

    def save_cleaned_data(self, filepath: str) -> None:
        """Save the cleaned data to a file.

        Parameters
        ----------
        filepath : str
            Path to save the data.
        """
        if filepath.endswith(".csv"):
            self.df.write_csv(filepath)
        elif filepath.endswith(".parquet"):
            self.df.write_parquet(filepath)
        elif filepath.endswith(".json"):
            self.df.write_json(filepath)
        else:
            self.df.write_csv(filepath)

        print(f"Saved cleaned data to {filepath}")
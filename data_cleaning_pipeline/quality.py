"""Data quality checks for cleaned data."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union


def data_quality_checks(
    df: pd.DataFrame,
    expected_types: Optional[Dict[str, str]] = None,
    required_columns: Optional[List[str]] = None,
    range_checks: Optional[Dict[str, Tuple[int, int]]] = None,
    id_column: Optional[str] = None,
    email_column: Optional[str] = None,
    report_only: bool = True,
) -> bool:
    """Run comprehensive data quality checks.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned DataFrame to validate.
    expected_types : dict, optional
        Expected data types for columns.
        Example: {'age': 'int64', 'price': 'float64'}
    required_columns : list, optional
        Columns that must exist in the DataFrame.
    range_checks : dict, optional
        Range validations: {'age': (0, 120)}
    id_column : str, optional
        Column name for unique identifier checks.
    email_column : str, optional
        Column name for email validation.
    report_only : bool, default True
        If True, print results and return boolean.
        If False, raise exceptions on failures.

    Returns
    -------
    bool
        True if all checks pass, False otherwise.

    Examples
    --------
    >>> df = pd.read_csv('cleaned_data.csv')
    >>> passed = data_quality_checks(df,
    ...     expected_types={'age': 'int64'},
    ...     required_columns=['id', 'name', 'email'],
    ...     range_checks={'age': (0, 120)})
    >>> if passed:
    ...     print("Data quality checks passed!")
    """
    report = []
    passed = True

    # Check 1: Required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            msg = f"❌ Required columns missing: {missing_cols}"
            report.append(msg)
            passed = False
        else:
            report.append("✅ All required columns present")

    # Check 2: Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_cols = missing[missing > 0]
        msg = f"⚠️  Missing values found:\n{missing_cols.to_string()}"
        report.append(msg)
        passed = False
    else:
        report.append("✅ No missing values")

    # Check 3: Duplicates (full row duplicates)
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        msg = f"⚠️  {duplicates} duplicate rows found"
        report.append(msg)
        passed = False
    else:
        report.append("✅ No duplicate rows")

    # Check 4: Data types
    if expected_types:
        for col, expected in expected_types.items():
            if col in df.columns:
                actual = str(df[col].dtype)
                # Handle equivalence (int64 ~ Int64, etc.)
                expected_normalized = expected.replace("int", "Int").replace("float", "Float")
                actual_normalized = actual.replace("int", "Int").replace("float", "Float")

                if expected_normalized not in actual_normalized and actual_normalized not in expected_normalized:
                    msg = f"⚠️  Column '{col}' has type {actual}, expected {expected}"
                    report.append(msg)
                    passed = False
                else:
                    report.append(f"✅ Column '{col}' has correct type: {actual}")

    # Check 5: Value ranges
    if range_checks:
        for col, (min_val, max_val) in range_checks.items():
            if col in df.columns:
                invalid = ((df[col] < min_val) | (df[col] > max_val)).sum()
                if invalid > 0:
                    msg = f"⚠️  {invalid} invalid {col} values outside range [{min_val}, {max_val}]"
                    report.append(msg)
                    passed = False
                else:
                    report.append(f"✅ Column '{col}' values within range [{min_val}, {max_val}]")

    # Check 6: Unique identifiers
    if id_column and id_column in df.columns:
        if df[id_column].duplicated().any():
            dups = df[id_column].duplicated().sum()
            msg = f"⚠️  {dups} duplicate IDs found in '{id_column}'"
            report.append(msg)
            passed = False
        else:
            report.append(f"✅ No duplicate IDs in '{id_column}'")

    # Check 7: Email consistency
    if email_column and email_column in df.columns:
        if df[email_column].dtype == object:
            invalid_email = (~df[email_column].str.contains("@", na=False)).sum()
            if invalid_email > 0:
                msg = f"⚠️  {invalid_email} invalid email addresses in '{email_column}'"
                report.append(msg)
                passed = False
            else:
                report.append(f"✅ All emails valid in '{email_column}'")

    # Check 8: Data shape consistency
    msg = f"📊 DataFrame shape: {df.shape[0]} rows x {df.shape[1]} columns"
    report.append(msg)

    # Print report
    print("\nData Quality Checks")
    print("=" * 50)
    for item in report:
        print(item)

    if passed:
        print("\n✅ All quality checks passed!")
    else:
        print("\n⚠️  Some quality checks failed - review above.")

    if not report_only and not passed:
        raise ValueError("Data quality checks failed!")

    return passed
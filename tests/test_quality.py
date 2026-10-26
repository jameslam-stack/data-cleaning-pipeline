"""Tests for data quality checks."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from data_cleaning_pipeline import data_quality_checks


def test_quality_checks_pass():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
    })
    result = data_quality_checks(
        df,
        expected_types={"id": "int64", "age": "int64"},
        required_columns=["id", "name", "age"],
        range_checks={"age": (0, 120)},
        id_column="id",
    )
    assert result is True
    print("PASS: test_quality_checks_pass")


def test_quality_checks_fail():
    df = pd.DataFrame({
        "id": [1, 2, 2],
        "name": ["Alice", "Bob", "Alice"],
        "age": [25, 30, 150],
    })
    result = data_quality_checks(
        df,
        range_checks={"age": (0, 120)},
        id_column="id",
    )
    assert result is False
    print("PASS: test_quality_checks_fail")


if __name__ == "__main__":
    test_quality_checks_pass()
    test_quality_checks_fail()
    print("All quality tests passed!")

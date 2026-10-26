"""Tests for the pandas cleaning pipeline."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from data_cleaning_pipeline import DataCleaningPipeline, auto_clean_dataframe, data_quality_checks


def test_pipeline_creation():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    p = DataCleaningPipeline(df)
    assert p.original_shape == (3, 2)
    print("PASS: test_pipeline_creation")


def test_remove_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    p = DataCleaningPipeline(df).remove_duplicates()
    assert len(p.get_cleaned_data()) == 2
    print("PASS: test_remove_duplicates")


def test_handle_missing_values():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    p = DataCleaningPipeline(df).handle_missing_values()
    assert p.get_cleaned_data()["a"].isnull().sum() == 0
    print("PASS: test_handle_missing_values")


def test_normalize_text():
    df = pd.DataFrame({"name": ["  Alice  ", "BOB"]})
    p = DataCleaningPipeline(df).normalize_text()
    cleaned = p.get_cleaned_data()
    assert cleaned["name"].tolist() == ["alice", "bob"]
    print("PASS: test_normalize_text")


def test_auto_clean():
    df = pd.DataFrame({
        "id": [1, 2, 1],
        "name": ["Alice", "Bob", "Alice"],
        "age": [25, np.nan, 25],
    })
    cleaned = auto_clean_dataframe(df)
    assert len(cleaned) == 2  # duplicate removed
    assert cleaned["age"].isnull().sum() == 0
    print("PASS: test_auto_clean")


if __name__ == "__main__":
    test_pipeline_creation()
    test_remove_duplicates()
    test_handle_missing_values()
    test_normalize_text()
    test_auto_clean()
    print("All pandas pipeline tests passed!")

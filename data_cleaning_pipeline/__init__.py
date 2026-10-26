"""Data cleaning pipeline generator for pandas, polars, and PySpark."""

from data_cleaning_pipeline.pandas_pipeline import DataCleaningPipeline
from data_cleaning_pipeline.quality import data_quality_checks
from data_cleaning_pipeline.auto_clean import auto_clean_dataframe

__version__ = "1.0.0"

__all__ = [
    "DataCleaningPipeline",
    "PolarsPipeline",
    "SparkPipeline",
    "data_quality_checks",
    "auto_clean_dataframe",
]


def __getattr__(name):
    """Lazy-load optional dependencies (polars, pyspark) on import."""
    if name == "PolarsPipeline":
        from data_cleaning_pipeline.polars_pipeline import PolarsPipeline
        return PolarsPipeline
    if name == "SparkPipeline":
        from data_cleaning_pipeline.spark_pipeline import SparkPipeline
        return SparkPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
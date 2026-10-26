# Data Cleaning Pipeline Generator

A comprehensive data cleaning pipeline generator for pandas, Polars, and PySpark with best practices for handling messy data.

## Overview

This repository implements data cleaning and preprocessing pipelines using pandas, polars, or PySpark with handling for:
- Missing values
- Duplicate rows
- Outliers (IQR and Z-score methods)
- Data type conversions
- Text normalization
- Data validation
- Auto-cleaning functionality

## Installation

Install directly from GitHub (clone the repo, then install in editable mode so
edits to the source are reflected immediately):

```bash
# 1. Clone the repo
git clone https://github.com/jameslam-stack/data-cleaning-pipeline.git
cd data-cleaning-pipeline

# 2. Install (editable mode — no PyPI required)
pip install -e .

# 3. (Optional) Install optional dependencies
pip install polars pyspark
```

> **Note:** This package is not on PyPI. Use the GitHub install above.
> Editable mode (`-e .`) means changes to the source files take effect without
> reinstalling — ideal for learning and development.

### Requirements

- Python 3.8+
- `pandas` and `numpy` (installed automatically)
- `polars` (optional, for the Polars pipeline)
- `pyspark` (optional, for the PySpark pipeline)

## Quick Start

```python
import pandas as pd
from data_cleaning_pipeline import DataCleaningPipeline, auto_clean_dataframe

# Load your data
df = pd.read_csv('your_data.csv')

# Using the pipeline class
pipeline = DataCleaningPipeline(df)
cleaned_df = (pipeline
    .remove_duplicates()
    .handle_missing_values()
    .fix_data_types()
    .remove_outliers()
    .normalize_text()
    .get_cleaned_data()
)

# Or use the quick auto-clean function
cleaned_df = auto_clean_dataframe(df)
```

## Features

| Feature | Description |
|---------|-------------|
| **Pandas Pipeline** | Full-fluent interface with logging |
| **Polars Pipeline** | Faster operations for large datasets |
| **PySpark Pipeline** | Big data support |
| **Auto Clean** | One-function cleaning with defaults |
| **Quality Checks** | Comprehensive validation |

## Usage Examples

See the `examples/` directory for complete examples:
- `pandas_example.py` - Pandas pipeline demo
- `polars_example.py` - Polars pipeline demo

## Project Structure

```
data-cleaning-pipeline/
├── data_cleaning_pipeline/     # Source package
│   ├── __init__.py             # Package exports
│   ├── pandas_pipeline.py      # Pandas implementation
│   ├── polars_pipeline.py      # Polars implementation
│   ├── spark_pipeline.py       # PySpark implementation
│   ├── auto_clean.py           # Auto-cleaning function
│   └── quality.py              # Data quality checks
├── tests/                      # Test suite
├── examples/                   # Example scripts
├── docs/                       # Documentation
├── README.md
├── pyproject.toml
└── LICENSE
```

## API Reference

### DataCleaningPipeline (Pandas)

```python
pipeline = DataCleaningPipeline(df)
pipeline.remove_duplicates()
pipeline.handle_missing_values(strategy='auto')
pipeline.fix_data_types()
pipeline.remove_outliers(method='iqr', threshold=1.5)
pipeline.normalize_text()
pipeline.validate_ranges({'age': (0, 120)})
cleaned = pipeline.get_cleaned_data()
```

### auto_clean_dataframe

```python
cleaned = auto_clean_dataframe(df, config={
    'remove_duplicates': True,
    'handle_missing': True,
    'normalize_text': True,
})
```

## License

MIT License - see LICENSE file.

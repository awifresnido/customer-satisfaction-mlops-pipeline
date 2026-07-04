"""Dataset validation utilities."""

import pandas as pd


class DataValidator:
    """Validate loaded datasets."""

    @staticmethod
    def validate(df: pd.DataFrame) -> None:
        """Validate a dataframe"""

        if df.empty:
            raise ValueError("Dataset is empty.")

        if df.columns.duplicated().any():
            raise ValueError("Duplicate column names detected.")

        if df.columns.empty:
            raise ValueError("Dataset contains no columns.")

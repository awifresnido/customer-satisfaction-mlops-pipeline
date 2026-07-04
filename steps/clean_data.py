"""ZenML step for data cleaning"""

import pandas as pd
from zenml import step

from customer_satisfaction.data.cleaner import DataCleaner


@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset.

    Args:
        df: Raw dataframe.

    Returns:
        Cleaned dataframe.
    """
    cleaner = DataCleaner()

    return cleaner.clean(df)

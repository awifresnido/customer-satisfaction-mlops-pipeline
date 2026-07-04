"""ZenML step for dataset splitting."""

from typing import Annotated

import pandas as pd
from zenml import step

from customer_satisfaction.data.splitter import DataSplitter


@step
def split_data(df: pd.DataFrame) -> tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Split the cleaned dataset.

    Args:
        df: Cleaned dataframe.

    Returns:
        X_train, X_test, y_train, y_test
    """

    splitter = DataSplitter()

    X_train, X_test, y_train, y_test = splitter.split(df)

    return (X_train, X_test, y_train, y_test)

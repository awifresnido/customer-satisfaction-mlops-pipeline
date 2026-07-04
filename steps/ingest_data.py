import pandas as pd
from zenml import step

from customer_satisfaction.data.loader import DataLoader


@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """Load the dataset."""

    loader = DataLoader(data_path)

    return loader.load()

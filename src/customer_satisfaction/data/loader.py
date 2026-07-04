"""Data loading utilities."""

from pathlib import Path

import pandas as pd

from customer_satisfaction.data.validator import DataValidator
from customer_satisfaction.logging.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Loads datasets from disk."""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load(self) -> pd.DataFrame:
        """Load a CSV dataset."""

        logger.info("Loading dataset from %s", self.data_path)

        if not self.data_path.exists():
            logger.error("Dataset not found: %s", self.data_path)
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        df = pd.read_csv(self.data_path)

        DataValidator.validate(df)

        logger.info("Dataset loaded successfully with shape %s", df.shape)

        return df

"""Dataset splitting utilities."""

import pandas as pd
from sklearn.model_selection import train_test_split

from customer_satisfaction.logging.logger import get_logger

logger = get_logger(__name__)


class DataSplitter:
    """Split the cleaned dataset into training and testing sets."""

    def split(
        self,
        df: pd.DataFrame,
        target_column: str = "review_score",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """Split the dataset into train and test sets.
        Args:
            df: Cleaned dataframe
            target_column: Target variable
            test_size: test_size
            random_state: seed number

        Returns:
            X_train, x_test, y_train, y_test
        """

        logger.info("Splitting dataset...")

        try:
            X = df.drop(columns=[target_column])
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            logger.info(
                "Dataset split complete." "Train: %s | Test: %s",
                X_train.shape,
                X_test.shape,
            )

            return (X_train, X_test, y_train, y_test)

        except Exception:
            logger.exception("Dataset splitting failed.")
            raise

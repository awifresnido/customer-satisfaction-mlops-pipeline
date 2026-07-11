"""Data cleaning utilities."""

import numpy as np
import pandas as pd

from customer_satisfaction.logging.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """Clean and preprocess the raw dataset."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform dataset cleaning and preprocessing.
        Args:
            df: raw dataframe
        Returns:
            cleaned dataframe
        """
        logger.info("Starting data cleaning...")

        try:
            df = df.drop(
                columns=[
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ]
            )

            numerical_columns = [
                "product_weight_g",
                "product_length_cm",
                "product_height_cm",
                "product_width_cm",
            ]

            for column in numerical_columns:
                df[column] = df[column].fillna(df[column].median())

            df["review_comment_message"] = df["review_comment_message"].fillna(
                "No review"
            )

            df = df.select_dtypes(include=[np.number])

            df = df.drop(columns=["customer_zip_code_prefix", "order_item_id"])

            logger.info("Data cleaning completed. Output shape: %s", df.shape)

            return df

        except Exception:
            logger.exception("Data cleaning failed.")
            raise

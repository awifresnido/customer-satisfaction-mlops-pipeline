"""Pipeline entry point."""

from customer_satisfaction.config.paths import RAW_DATA
from pipelines.data_pipeline import data_pipeline

if __name__ == "__main__":
    data_pipeline(
        data_path=str(RAW_DATA),
    )

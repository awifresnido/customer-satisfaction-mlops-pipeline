"""Pipeline entry point."""

import argparse

from customer_satisfaction.config.paths import RAW_DATA
from customer_satisfaction.models.registry import MODEL_REGISTRY
from pipelines.data_pipeline import data_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        choices=list(MODEL_REGISTRY),
        default="random_forest",
        help="Which registered model to train.",
    )
    args = parser.parse_args()

    data_pipeline(
        data_path=str(RAW_DATA),
        model_type=args.model_type,
    )

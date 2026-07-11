"""ZenML data preparation and training pipeline"""

from typing import Any

from zenml import pipeline

from steps.clean_data import clean_data
from steps.evaluate_model import evaluate_model
from steps.ingest_data import ingest_data
from steps.split_data import split_data
from steps.train_model import train_model


@pipeline
def data_pipeline(
    data_path: str,
    model_type: str = "random_forest",
    hyperparameters: dict[str, Any] | None = None,
):
    """Execute the end-to-end training pipeline"""

    raw_df = ingest_data(data_path=data_path)
    cleaned_df = clean_data(df=raw_df)
    X_train, X_test, y_train, y_test = split_data(df=cleaned_df)
    model = train_model(
        X_train=X_train,
        y_train=y_train,
        model_type=model_type,
        hyperparameters=hyperparameters,
    )
    metrics = evaluate_model(model=model, X_test=X_test, y_test=y_test)

    return model, metrics

"""ZenML step for model training."""

from typing import Any

import mlflow
import mlflow.lightgbm as mlflow_lightgbm
import mlflow.sklearn as mlflow_sklearn
import mlflow.xgboost as mlflow_xgboost
from zenml import step

from customer_satisfaction.models.trainer import ModelTrainer

_MLFLOW_LOGGERS = {
    "sklearn": mlflow_sklearn.log_model,
    "xgboost": mlflow_xgboost.log_model,
    "lightgbm": mlflow_lightgbm.log_model,
}


@step(experiment_tracker=True, enable_cache=False)
def train_model(
    X_train,
    y_train,
    model_type: str = "random_forest",
    hyperparameters: dict[str, Any] | None = None,
):
    """Train a regression model.

    Args:
        X_train: training features
        y_train: training targets
        model_type: key into the model registry, e.g. "random_forest",
            "linear_regression", "extra_trees", "lightgbm", "xgboost"
        hyperparameters: overrides for the model's default hyperparameters

    Returns:
        Trained model
    """
    trainer = ModelTrainer(model_type=model_type, **(hyperparameters or {}))

    model = trainer.train(X_train=X_train, y_train=y_train)

    mlflow.set_tag("model_type", model_type)
    for name, value in trainer.params.items():
        mlflow.log_param(name, value)

    log_model = _MLFLOW_LOGGERS[trainer.mlflow_flavor]
    log_model(model, artifact_path="model")

    return model

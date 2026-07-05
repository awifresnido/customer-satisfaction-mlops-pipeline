"""ZenML step for model training."""

import mlflow
import mlflow.sklearn as mlflow_sklearn
from sklearn.ensemble import RandomForestRegressor
from zenml import step

from customer_satisfaction.models.trainer import RandomForestTrainer


@step(experiment_tracker=True, enable_cache=False)
def train_model(X_train, y_train) -> RandomForestRegressor:
    """Train a Random Forest model.
    Args:
        X_train: training features
        y_train: training targets

    Returns:
        Trained RandomForestRegressor
    """
    trainer = RandomForestTrainer()

    model = trainer.train(X_train=X_train, y_train=y_train)

    mlflow.set_tag("model_type", type(model).__name__)
    mlflow.log_param("n_estimators", trainer.n_estimators)
    mlflow.log_param("random_state", trainer.random_state)

    mlflow_sklearn.log_model(sk_model=model, artifact_path="model")

    return model

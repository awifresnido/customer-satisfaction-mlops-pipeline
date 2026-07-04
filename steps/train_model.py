"""ZenML step for model training."""

from sklearn.ensemble import RandomForestRegressor
from zenml import step

from customer_satisfaction.models.trainer import RandomForestTrainer


@step
def train_model(X_train, y_train) -> RandomForestRegressor:
    """Train a Random Forest model.
    Args:
        X_train: training features
        y_train: training targets

    Returns:
        Trained RandomForestRegressor
    """
    trainer = RandomForestTrainer()

    return trainer.train(X_train=X_train, y_train=y_train)

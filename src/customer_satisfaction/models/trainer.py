"""Model training utilities"""

from typing import Any

from customer_satisfaction.logging.logger import get_logger
from customer_satisfaction.models.registry import MODEL_REGISTRY

logger = get_logger(__name__)


class ModelTrainer:
    """Train a regression model looked up from the model registry."""

    def __init__(self, model_type: str, **hyperparameters: Any):
        if model_type not in MODEL_REGISTRY:
            available = ", ".join(MODEL_REGISTRY)
            raise ValueError(
                f"Unknown model_type '{model_type}'. Available: {available}"
            )

        spec = MODEL_REGISTRY[model_type]

        self.model_type = model_type
        self.mlflow_flavor = spec.mlflow_flavor
        self.params = {**spec.default_params, **hyperparameters}
        self._model_class = spec.model_class

    def train(self, X_train, y_train):
        """Train the model.
        Args:
            X_train: training features
            y_train: training targets

        Returns:
            Trained model instance
        """

        logger.info(
            "Initializing %s model with params: %s", self.model_type, self.params
        )
        model = self._model_class(**self.params)

        logger.info("Training model...")
        model.fit(X_train, y_train)

        logger.info("Training complete.")

        return model

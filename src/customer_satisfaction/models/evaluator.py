"""Model evaluation utilities"""

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from customer_satisfaction.logging.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluate trained machine learning model."""

    def evaluate(self, model, X_test, y_test) -> dict:
        """Evaluate a trained model.
        Args:
            model: trained model
            X_test: test features
            y_test: test targets

        Returns:
            Dictionary containing evaluation metrics.
        """
        logger.info("Evaluating model...")

        predictions = model.predict(X_test)

        metrics = {
            "mae": mean_absolute_error(y_test, predictions),
            "rmse": root_mean_squared_error(y_test, predictions),
            "r2": r2_score(y_test, predictions),
        }

        logger.info("Evaluation complete: %s", metrics)

        return metrics

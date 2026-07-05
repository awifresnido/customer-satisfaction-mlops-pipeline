"""ZenML step for model evaluation."""

import mlflow
from zenml import step

from customer_satisfaction.models.evaluator import ModelEvaluator


@step(experiment_tracker=True, enable_cache=False)
def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate a trained model.
    Args:
        model: trained model
        X_test: test features
        y_test: test labels

    Returns:
        Dictionary of evaluation metrics
    """
    evaluator = ModelEvaluator()

    metrics = evaluator.evaluate(model=model, X_test=X_test, y_test=y_test)

    mlflow.log_metrics(metrics)

    return metrics

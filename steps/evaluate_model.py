"""ZenML step for model evaluation."""

from zenml import step

from customer_satisfaction.models.evaluator import ModelEvaluator


@step
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

    return evaluator.evaluate(model=model, X_test=X_test, y_test=y_test)

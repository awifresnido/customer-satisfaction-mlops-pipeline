from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Represents the evaluation metrics of a trained model"""

    mae: float
    rmse: float
    r2: float

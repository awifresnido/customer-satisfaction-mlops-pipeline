from dataclasses import dataclass

from customer_satisfaction.domain.evaluation_result import EvaluationResult
from customer_satisfaction.domain.training_result import TrainingResult


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Represents the complete outcome of an ML experiment."""

    training: TrainingResult
    evaluation: EvaluationResult

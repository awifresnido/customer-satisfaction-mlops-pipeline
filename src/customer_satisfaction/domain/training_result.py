from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """
    Container for the output of a model training run.

    This object decouples model training from downstream concerns
    such as experiment tracking and model registration.
    """

    model: Any
    parameters: dict[str, Any]

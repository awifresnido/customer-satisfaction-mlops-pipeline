from abc import ABC, abstractmethod

from customer_satisfaction.domain.experiment_result import ExperimentResult


class ExperimentTracker(ABC):
    """Abstract interface for experiment tracking backends."""

    @abstractmethod
    def track_experiment(self, experiment: ExperimentResult) -> None:
        """Persist an experiment using the configured tracking backend."""

        raise NotImplementedError

"""Registry of supported regression models.

Adding a new model only requires a new entry here - no other file needs
to change for training, mlflow logging, or evaluation to pick it up.
"""

from dataclasses import dataclass, field
from typing import Any

from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to instantiate, train and log a model."""

    model_class: type
    mlflow_flavor: str
    default_params: dict[str, Any] = field(default_factory=dict)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "random_forest": ModelSpec(
        model_class=RandomForestRegressor,
        mlflow_flavor="sklearn",
        default_params={"n_estimators": 100, "random_state": 42},
    ),
    "linear_regression": ModelSpec(
        model_class=LinearRegression,
        mlflow_flavor="sklearn",
    ),
    "extra_trees": ModelSpec(
        model_class=ExtraTreesRegressor,
        mlflow_flavor="sklearn",
        default_params={"n_estimators": 100, "random_state": 42},
    ),
    "lightgbm": ModelSpec(
        model_class=LGBMRegressor,
        mlflow_flavor="lightgbm",
        default_params={"random_state": 42},
    ),
    "xgboost": ModelSpec(
        model_class=XGBRegressor,
        mlflow_flavor="xgboost",
        default_params={"random_state": 42},
    ),
}

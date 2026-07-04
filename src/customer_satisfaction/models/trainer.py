"""Model training utilities"""

from sklearn.ensemble import RandomForestRegressor

from customer_satisfaction.logging.logger import get_logger

logger = get_logger(__name__)


class RandomForestTrainer:
    """Train a Random Forest regression model"""

    def __init__(self, random_state: int = 42, n_estimators: int = 100):

        self.random_state = random_state
        self.n_estimators = n_estimators

    def train(self, X_train, y_train):
        """Train the model.
        Args:
            X_train: training features
            y_train: training targets

        Returns:
            Trained RandomForestRegressor
        """

        logger.info("Initializing Random Forest model...")
        model = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=self.random_state
        )

        logger.info("Training model...")
        model.fit(X_train, y_train)

        logger.info("Training complete.")

        return model

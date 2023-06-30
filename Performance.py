import time  # For measuring time
import numpy as np

# from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# from sklearn.preprocessing import MinMaxScaler


class Performance:
    def __init__(self):
        pass

    def k_fold(
        self,
        model,
        data: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.3,
        k: int = 10,
        n_dim: int = 2,
    ) -> list[float]:
        """
        Returns the average performance of a model
        using k-fold cross validation

        avg(accuracy_score, f1_score, precision_score, recall_score, time)
        """
        # Parameters
        avg_accuracy_score = 0
        avg_f1_score = 0
        avg_precision_score = 0
        avg_recall = 0
        avg_time = 0

        # Normalization
        scaler = MinMaxScaler()
        scaler.fit_transform(data)

        # Start time
        start = time.time()

        for _ in range(k):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                data, labels, test_size=test_size
            )

            # Train
            model.fit(X_train, y_train)

            # Predict
            predictions = model.predict(X_test)

            # Evaluate
            result = model.evaluate(y_test, predictions)
            # Measuring performance
            avg_accuracy_score += accuracy_score(y_test, predictions)
            avg_f1_score += f1_score(
                y_test, predictions, labels=np.unique(labels), average="micro"
            )
            avg_precision_score += precision_score(
                y_test, predictions, average="micro"
            )
            avg_recall += recall_score(y_test, predictions, average="micro")
            avg_time += time.time() - start

        # Return average performance
        # result = model.evaluate(y_test, predictions) + [avg_time / k]
        result = np.hstack((model.evaluate(y_test, predictions), avg_time / k))
        return np.round(result, n_dim)

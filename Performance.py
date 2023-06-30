import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


class Performance:
    def __init__(self):
        pass

    def k_fold(model, data: np.ndarray, labels, test_size=0.3, k=10):
        # Parameters
        avg_accuracy_score = 0
        avg_f1_score = 0
        avg_precision_score = 0
        avg_recall = 0

        # Scaler
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)

        for _ in range(k):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                data, labels, test_size, random_state=42
            )

            # Train
            model.fit(X_train, y_train)

            # Predict
            predictions = model.predict(X_test)

            # Measuring performance
            avg_accuracy_score += accuracy_score(predictions, y_test)
            avg_f1_score += f1_score(predictions, y_test)
            avg_precision_score += precision_score(predictions, y_test)
            avg_recall += recall_score(predictions, y_test)

        # Return average performance
        return (
            avg_accuracy_score / k,
            avg_f1_score / k,
            avg_precision_score / k,
            avg_recall / k,
        )

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from kernel import kernel
from show import show_data


class knn:
    def __init__(self, k: int, kernel_type: kernel = None):
        self.k = k
        self.kernel_type = kernel_type
        if self.kernel_type is None:
            self.kernel_type = kernel()

    def fit(self, data, labels):
        self.data = data
        self.labels = labels

    def predict(self, data):
        predictions = []
        for i in range(len(data)):
            distances = [
                self.kernel_type.distance(data[i], self.data[j])
                for j in range(len(self.data))
            ]
            k_nearest = np.argsort(distances)[: self.k]
            k_nearest_labels = [self.labels[j] for j in k_nearest]
            predictions.append(
                max(set(k_nearest_labels), key=k_nearest_labels.count)
            )
        return predictions


def sample():
    # Data
    data, label = datasets.make_moons(n_samples=100, shuffle=True)
    data, label = datasets.make_circles(n_samples=100, shuffle=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data, label, test_size=0.3, random_state=42
    )

    # Train
    clf = knn(k=2, kernel_type=kernel())
    clf.fit(X_train, y_train)

    # Predict
    predictions = clf.predict(X_test)
    print(predictions)

    # Show test data
    show_data.show(None, X_test, predictions)


sample()

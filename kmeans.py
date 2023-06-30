import numpy as np
from kernel import kernel
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class kmeans:
    def __init__(
        self, k: int, max_iter: int = 100, kernel_type: kernel = None
    ):
        self.k = k
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        if self.kernel_type is not None:
            self.kernel_type = kernel_type
        self.centroids = {}

    def fit(self, data, labels, kernel_type: kernel = None):
        if self.kernel_type is None and kernel_type is None:
            KeyError("Kernel type not found")
        if self.kernel_type is None:
            self.kernel_type = kernel_type
        if kernel_type is not None and self.kernel_type is not None:
            KeyError("Kernel type already defined")

        # Generate the k random number
        random_index = np.random.choice(len(data), self.k, replace=False)

        # Initialize the centroids
        self.centroids = {}
        for n_centroid in range(self.k):
            self.centroids[n_centroid] = data[random_index[n_centroid]]

        for _ in range(self.max_iter):
            self.clusterings = {}
            for each_centroid in range(self.k):
                self.clusterings[each_centroid] = []

            # calculate the distance between each point and the centroids
            for point in data:
                distances = [
                    self.kernel_type.distance(point, self.centroids[centroid])
                    for centroid in self.centroids
                ]
                # find the closest centroid
                classification = distances.index(min(distances))
                # add the point to the closest cluster
                self.clusterings[classification].append(point)

            # calculate the new centroids
            prev_centroids = dict(self.centroids)
            for classification in self.clusterings:
                self.centroids[classification] = np.average(
                    self.clusterings[classification], axis=0
                )

            # check if the centroids have moved
            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if (
                    np.sum(
                        (current_centroid - original_centroid)
                        / original_centroid
                        * 100.0
                    )
                    > 0.001
                ):
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [
            self.kernel_type.distance(data, self.centroids[centroid])
            for centroid in self.centroids
        ]
        classification = distances.index(min(distances))

        return classification


def sample():
    # Data
    data, label = datasets.make_moons(n_samples=100, shuffle=True)
    data, label = datasets.make_circles(n_samples=100, shuffle=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data, label, test_size=0.3, random_state=42
    )

    # Train
    clf = kmeans(k=2, max_iter=100, kernel_type=kernel(kernel_type="RBF"))
    clf.fit(X_train)

    # Predict
    predictions = []
    for i in range(len(X_test)):
        predictions.append(clf.predict(X_test[i]))

    # Accuracy using confusion matrix
    conf_mat = confusion_matrix(y_test, predictions)
    print(conf_mat)


# sample()

import numpy as np
from kernel import kernel
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class kmeans:
    def __init__(
        self,
        k: int,
        max_iter: int = 100,
        kernel_type: kernel = None,
        n_dim: int = 3,
    ):
        self.k = k
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        if self.kernel_type is not None:
            self.kernel_type = kernel_type
        self.centroids = {}
        self.n_dim = n_dim

    def fit(self, train_data_s, labels, kernel_type: kernel = None):
        if self.kernel_type is None and kernel_type is None:
            KeyError("Kernel type not found")
        if self.kernel_type is None:
            self.kernel_type = kernel_type
        if kernel_type is not None and self.kernel_type is not None:
            KeyError("Kernel type already defined")

        # Generate the k random number
        random_index = np.random.choice(
            len(train_data_s), self.k, replace=False
        )

        # Initialize the centroids
        self.centroids = {}
        for n_centroid in range(self.k):
            self.centroids[n_centroid] = train_data_s[random_index[n_centroid]]

        for _ in range(self.max_iter):
            self.clusterings = {}
            for each_centroid in range(self.k):
                self.clusterings[each_centroid] = []

            # calculate the distance between each point and the centroids
            for train_data in train_data_s:
                distances = [
                    self.kernel_type.distance(
                        train_data, self.centroids[centroid]
                    )
                    for centroid in self.centroids
                ]
                # find the closest centroid
                close_centroid_id = distances.index(min(distances))
                # add the train_data to the closest cluster
                self.clusterings[close_centroid_id].append(train_data)

            # calculate the new centroids
            prev_centroids = dict(self.centroids)
            for cluster in self.clusterings:
                self.centroids[cluster] = np.average(
                    self.clusterings[cluster], axis=0
                )

            # check if the centroids have moved
            # By checking if the centroids have moved
            # we can stop the algorithm if the centroids
            # have not moved(small change)
            optimized = True
            for centroid in self.centroids:
                prev_centroid = prev_centroids[centroid]
                current_centroid = self.centroids[centroid]

                # check if the centroids have moved more than 0.001%
                precent_error = np.sum(
                    abs(current_centroid - prev_centroid)
                    / prev_centroid
                    * 100.0
                )

                if precent_error > 0.001:
                    optimized = False

            if optimized:
                break

    def predict(self, test_data_s):
        distances = np.array(
            [
                [
                    self.kernel_type.distance(centroid, test_data)
                    for centroid in self.centroids.values()
                ]
                for test_data in test_data_s
            ]
        )

        # Choose closest cluster for each test point
        clusters = np.argmin(distances, axis=1)

        return clusters

    def evaluate(self, y_test: list, predictions: list):
        # Accuracy using confusion matrix
        conf_mat = confusion_matrix(y_test, predictions)

        # Reassign the labels
        new_labels = np.argmax(conf_mat, axis=1)

        # rearrange the confusion matrix
        conf_mat = conf_mat[new_labels]

        TP, FP, FN, TN = (
            conf_mat[0][0],
            conf_mat[0][1],
            conf_mat[1][0],
            conf_mat[1][1],
        )
        # Accuracy
        accuracy = (TP + TN) / (TP + FP + FN + TN)

        # Precision
        precision = TP / (TP + FP)

        # Recall
        recall = TP / (TP + FN)

        # F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall)

        return np.round([accuracy, precision, recall, f1_score], self.n_dim)


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


# sample()

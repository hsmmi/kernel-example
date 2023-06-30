from matplotlib import pyplot as plt
import numpy as np


class show_data:
    def __init__(self) -> None:
        pass

    def show(self, data: np.ndarray, labels: np.ndarray, title=None):
        # Plot the data
        if len(data.shape) == 1 or data.shape[1] == 1:
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            plt.scatter(
                data[:, 0], np.zeros(len(data)), c=labels, cmap="viridis"
            )
            plt.title(title)
            plt.show()
        else:
            plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="viridis")
            plt.title(title)
            plt.show()

    def plot_clusters(self, X_train, X_test, kernel_kmeans, dataset_name):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(
            f"{dataset_name} Kernel K-means Clustering",
            fontsize=12,
            fontweight="bold",
        )

        # Plot train clusters
        train_clusters = kernel_kmeans.predict(X_train)
        unique_clusters_train = np.unique(train_clusters)
        colors_train = ["red", "blue", "green", "purple", "orange", "yellow"]
        for cluster_id in unique_clusters_train:
            cluster_data = X_train[train_clusters == cluster_id]
            axes[0].scatter(
                cluster_data[:, 0],
                cluster_data[:, 1],
                s=3.5,
                color=colors_train[cluster_id],
            )

        # convert dict to np.ndarray
        # centroids = np.array(list(kernel_kmeans.centroids.values()))

        axes[0].scatter(
            kernel_kmeans.centroids[0],
            kernel_kmeans.centroids[1],
            c="black",
            marker="x",
            label="Centroids",
        )
        axes[0].set_title("Train Clusters", fontsize=10)

        # Plot test clusters
        test_clusters = kernel_kmeans.predict(X_test)
        unique_clusters_test = np.unique(test_clusters)
        colors_test = [
            "lightcoral",
            "lightblue",
            "lightgreen",
            "plum",
            "gold",
            "palegreen",
        ]
        for cluster_id in unique_clusters_test:
            cluster_data = X_test[test_clusters == cluster_id]
            axes[1].scatter(
                cluster_data[:, 0],
                cluster_data[:, 1],
                s=3.5,
                color=colors_test[cluster_id],
            )
        axes[1].scatter(
            kernel_kmeans.centroids[0],
            kernel_kmeans.centroids[1],
            c="black",
            marker="x",
            label="Centroids",
        )
        axes[1].set_title("Test Clusters", fontsize=10)
        plt.show()

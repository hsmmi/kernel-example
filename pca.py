import numpy as np
from sklearn import datasets
from kernel import kernel
from show import show_data


class pca:
    def __init__(self, n_components, kernel_type: kernel = None):
        self.n_components = n_components
        self.eig_vec_sc = None
        self.kernel_type = kernel_type
        if self.kernel_type is None:
            self.kernel_type = "simple"

    def fit(self, data):
        # Create kernel matrix
        kernel_matrix = np.array(
            [
                [
                    self.kernel_type.distance(data[i], data[j])
                    for i in range(len(data))
                ]
                for j in range(len(data))
            ]
        )

        # Scatter matrix
        scatter_matrix = np.array(
            [
                [
                    np.dot(
                        kernel_matrix[i] - np.mean(kernel_matrix[i]),
                        kernel_matrix[j] - np.mean(kernel_matrix[j]),
                    )
                    for i in range(len(kernel_matrix))
                ]
                for j in range(len(kernel_matrix))
            ]
        )

        # Eigenvalues and eigenvectors
        eig_val_sc, eig_vec_sc = np.linalg.eigh(scatter_matrix)

        # Sort the eigenvectors and eigenvalues in descending order
        eig_vec_sc = eig_vec_sc[:, np.argsort(eig_val_sc)[::-1]]
        eig_val_sc = eig_val_sc[np.argsort(eig_val_sc)[::-1]]

        # Select the first n eigenvectors
        self.eig_vec_sc = eig_vec_sc[:, : self.n_components]

        # Project the data
        return np.dot(kernel_matrix, self.eig_vec_sc)

    def transform(self, data):
        # Transform the data
        return np.dot(data, self.eig_vec_sc)


def sample():
    # Data
    data, label = datasets.make_moons(n_samples=100, shuffle=True)
    data, label = datasets.make_circles(n_samples=100, shuffle=True)

    # Train
    clf = pca(n_components=1, kernel_type=kernel(kernel_type="RBF"))
    data = clf.fit(data)

    show_data().show(data, label)


sample()

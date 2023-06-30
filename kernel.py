import numpy as np


class kernel:
    def __init__(self, kernel_type: str = "simple") -> None:
        self.kernel_type = kernel_type

    def simple_kernel(self, x: np.ndarray, xp: np.ndarray):
        return self.euclidean_distance(x, xp)

    def linear_kernel(self, x: np.ndarray, xp: np.ndarray):
        return x @ xp.T

    def polynomial_kernel(self, x: np.ndarray, xp: np.ndarray, d: int):
        return (self.linear_kernel(x, xp)) ** d

    def kernel_RBF(self, x: np.ndarray, xp: np.ndarray, sigma: float):
        return np.exp(-1 * self.euclidean_distance(x, xp) / (2 * sigma**2))

    def euclidean_distance(self, x: np.ndarray, xp: np.ndarray):
        return np.linalg.norm(x - xp)

    def distance(self, x: np.ndarray, xp: np.ndarray, sigma: float = 1.0):
        if self.kernel_type == "simple":
            return self.simple_kernel(x, xp)

        elif self.kernel_type == "linear":
            return (
                self.linear_kernel(x, x)
                + self.linear_kernel(xp, xp)
                - 2 * self.linear_kernel(x, xp)
            )

        elif self.kernel_type == "polynomial":
            return (
                self.polynomial_kernel(x, x)
                + self.polynomial_kernel(xp, xp)
                - 2 * self.polynomial_kernel(x, xp)
            )

        elif self.kernel_type == "RBF":
            return (
                self.kernel_RBF(x, x)
                + self.kernel_RBF(xp, xp)
                - 2 * self.kernel_RBF(x, xp)
            )
        else:
            raise ValueError("Kernel type not found")

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

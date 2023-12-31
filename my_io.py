import numpy as np
import pandas as pd


class my_io:
    def __init__(self, path: str = None) -> None:
        self.path = path

    def read_csv(self, path: str = None) -> list[np.ndarray, np.ndarray]:
        if self.path is None:
            self.path = path
        if self.path is None:
            raise ValueError("Path not found")

        # csv to ndarray
        data, labels = [], []
        df = pd.read_csv(self.path)
        for row in df.values:
            data.append(row[:-1])
            labels.append(row[-1])

        data = np.array(data)
        labels = np.array(labels)

        # Y_test is not nan
        labels = labels[~np.isnan(labels)]
        if len(labels) == 0:
            print("No labels found")

        return data, labels

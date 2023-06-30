import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold


class KNN:
    def __init__(self, k, data_train, label_train, data_test):
        self.k = k
        self.data_train = data_train
        self.data_test = data_test
        self.label_train = label_train

    # implementing the euclidean distance
    def euclidean_distance(self, x, xp):
        return np.sqrt(np.sum(np.square(x - xp)))

    def linear_kernel(self, x, xp):
        return x @ xp.T

    def kernel_RBF(self, x, xp, sigma):
        #x, xp = np.float64(x), np.float64(xp)
        return np.exp(-1 * self.euclidean_distance(x, xp) ** 2 / (2 * sigma ** 2))

    def polynomial_kernel(self, x, xp, d):
        return (self.linear_kernel(x, xp)) ** d

    # implementing the K-nearest neighbor
    def simple_NN(self, kernel_type="simple", d=2, sigma=0.0001):
        label_test = []
        for i in range(self.data_test.shape[0]):
            dis = []
            ind_dis = []
            for j in range(self.data_train.shape[0]):

                if kernel_type == "simple":
                    # find distance of each test data point with all the training datas
                    dis.append(self.euclidean_distance(self.data_train[j], self.data_test[i]))

                if kernel_type == "linear":
                    # apply linear kernel to classifying
                    dis.append(self.linear_kernel(self.data_train[j], self.data_train[j]) +
                               self.linear_kernel(self.data_test[i], self.data_test[i]) -
                               2 * self.linear_kernel(self.data_test[i], self.data_train[j]))

                if kernel_type == "RBF":
                    # apply RBF kernel to classifying
                    dis.append(self.kernel_RBF(self.data_train[j], self.data_train[j], sigma=sigma) +
                               self.kernel_RBF(self.data_test[i], self.data_test[i], sigma=sigma) -
                               2 * self.kernel_RBF(self.data_test[i], self.data_train[j], sigma=sigma))

                if kernel_type == "polynomial":
                    # apply polynomial kernel to classifying
                    dis.append(self.polynomial_kernel(self.data_train[j], self.data_train[j], d=d) +
                               self.polynomial_kernel(self.data_test[i], self.data_test[i], d=d) -
                               2 * self.polynomial_kernel(self.data_test[i], self.data_train[j], d=d))

                # append index of each training point
                ind_dis.append(j)

            # now after finding the distances we want to sort them
            dis_sorted, ind_dis_sorted = zip(*sorted(zip(dis, ind_dis)))

            # collect k nearest neighbors
            ind_k = ind_dis_sorted[:self.k]

            # find labels of nearest examples
            label_k = [self.label_train[index] for index in ind_k]

            # find the most frequent value in this array
            f_label = np.bincount(label_k).argmax()

            # add this label to our label list
            label_test.append(f_label)
        return label_test


data_path = 'D:\Biomedical master courses\Machine learning\HW\HW5_ML\HW5_ML\Datasets\Datasets\Glass.txt'
data = np.loadtxt(data_path)


data_main = data[:, :-1]
scalar = MinMaxScaler()
data_main = scalar.fit_transform(data_main)
label_main = data[:, -1]

train_x, test_x, train_y, test_y = train_test_split(data_main, label_main, test_size=0.3, random_state=42)
k = 1
#KNN_classifier = KNN(k=1, data_train=train_x, label_train=train_y, data_test=test_x)

#label_pred = KNN_classifier.simple_NN(kernel_type="RBF", d=2)

#print(accuracy_score(test_y, label_pred))

# find RBF sigma parameter
gamma_values = [0.0001, 0.001, 0.005, 0.1, 0.5, 1, 5, 10]
best_gamma = None
best_accuracy = 0

# Define the number of folds for cross-validation
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Iterate over gamma values
for gamma in gamma_values:
    accuracies = []

    # Perform cross-validation
    for train_index, val_index in kf.split(data_main):  # X is your feature data
        X_train, X_val = data_main[train_index], data_main[val_index]
        y_train, y_val = label_main[train_index], label_main[val_index]

        # Create and fit the SVM classifier
        KNN_classifier = KNN(k=1, data_train=X_train, label_train=y_train, data_test=X_val)
        y_pred = KNN_classifier.simple_NN(kernel_type="RBF", d=2)

        accuracy = accuracy_score(y_val, y_pred)
        accuracies.append(accuracy)

    # Calculate the average accuracy across folds
    mean_accuracy = np.mean(accuracies)

    # Check if the current gamma value is the best so far
    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_gamma = gamma

print("Best gamma value:", best_gamma, "best accuracy with this RBF:", best_accuracy)



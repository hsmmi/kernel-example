from sklearn import datasets
from Performance import Performance
from kernel import kernel
from kmeans import kmeans

# from knn import knn
from my_io import my_io


ds_path = "./Datasets/"
ds_file = [
    "Wine.txt",
    "Glass.txt",
    "BreastTissue.txt",
    "Diabetes.txt",
    "Sonar.txt",
    "Ionosphere.txt",
]
# for file in ds_file:
# path = ds_path + file
# print("Dataset: ", file)

# data, label = my_io(path).read_csv()

ds_file = [
    datasets.make_moons(n_samples=1000, shuffle=True),
    datasets.make_circles(n_samples=1000, shuffle=True),
]

for data, label in ds_file:
    model = kmeans(k=2, kernel_type=kernel("linear"))
    [
        avg_accuracy_score,
        avg_f1_score,
        avg_precision_score,
        avg_recall,
        avg_time,
    ] = Performance.k_fold(
        Performance(),
        model=model,
        data=data,
        labels=label,
        test_size=0.3,
        k=100,
    )

    # using confusion matrix for performance metrics
    print("Accuracy: ", avg_accuracy_score)
    print("F1 score: ", avg_f1_score)
    print("Precision: ", avg_precision_score)
    print("Recall: ", avg_recall)
    print("Time: ", avg_time)
    print(f"{'='*50}\n")

    # from show import show_data

    # show_data().plot_clusters(data, data, model, "data")

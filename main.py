from collections import defaultdict

from EvaluationMetrics import EvaluationMetrics
from Perceptron import Perceptron
import matplotlib.pyplot as plt
import random

def readIris():
    f = open('iris.csv')
    content = f.read()
    f.close()
    return content.strip().split('\n')[1:]




class MyVector:
    def __init__(self, values: list, label: int):
        self.values = values
        self.label = label

    def is_class_name_equal(self, another_vector) -> bool:
        return self.label == another_vector.label

class PrepareDataset:
    train_dataset: list = []
    test_dataset: list = []

    @staticmethod
    def trainTestSplit(dataset: list, columns: int,labelForOne: str):

        PrepareDataset.train_dataset.clear()
        PrepareDataset.test_dataset.clear()

        # Group rows by class label
        class_map = defaultdict(list)
        for line in dataset:
            info = line.split(",")
            label = info[columns - 1].strip()
            num = 1 if label == labelForOne else 0


            values = [float(info[j]) for j in range(len(info) - 1)]
            class_map[num].append(MyVector(values, num))

        for key, vectors in class_map.items():
            n_train = int(len(vectors) * 0.7)

            PrepareDataset.train_dataset.extend(vectors[:n_train])
            PrepareDataset.test_dataset.extend(vectors[n_train:])

        print(f"Train size: {len(PrepareDataset.train_dataset)}")
        print(f"Test size: {len(PrepareDataset.test_dataset)}")



    @staticmethod
    def get_train_dataset() -> list:
        return PrepareDataset.train_dataset

    @staticmethod
    def get_test_dataset() -> list:
        return PrepareDataset.test_dataset


def trainAndTest(showResults = True):
    PrepareDataset.trainTestSplit(readIris(), columns=5, labelForOne="setosa")
    train = PrepareDataset.get_train_dataset()
    inputs = [v.values for v in train]
    labels = [v.label for v in train]
    p = Perceptron(dimension=4, threshold=0.5, alpha=0.1, beta=0.1)
    p.train(inputs, labels, epochs=1000,showResults = showResults)

    test = PrepareDataset.get_test_dataset()
    test_inputs = [v.values for v in test]
    test_labels = [v.label for v in test]
    predicted = [p.predict(v) for v in test_inputs]
    accuracy, values = EvaluationMetrics.measureAccuracy(predicted, test_labels)
    print(f"Test accuracy: {accuracy:.1f}% ({values})")
    return p



def fourDim():
    train = PrepareDataset.get_train_dataset()
    inputs = [v.values for v in train]
    labels = [v.label for v in train]

    test = PrepareDataset.get_test_dataset()
    test_inputs = [v.values for v in test]
    test_labels = [v.label for v in test]

    p = trainAndTest()

    x_coords = [v[2] for v in test_inputs]
    y_coords = [v[3] for v in test_inputs]


    x_setosa = [x_coords[i] for i in range(len(test_labels)) if test_labels[i] == 1]
    y_setosa = [y_coords[i] for i in range(len(test_labels)) if test_labels[i] == 1]

    x_versi = [x_coords[i] for i in range(len(test_labels)) if test_labels[i] == 0]
    y_versi = [y_coords[i] for i in range(len(test_labels)) if test_labels[i] == 0]


    plt.scatter(x_setosa, y_setosa, c='red', label='Setosa')
    plt.scatter(x_versi, y_versi, c='blue', label='Versicolor')


    avg_x0 = sum(v[0] for v in inputs) / len(inputs)
    avg_x1 = sum(v[1] for v in inputs) / len(inputs)


    eff_threshold = p.threshold - (p.weights[0] * avg_x0 + p.weights[1] * avg_x1)

    w = p.weights
    x_min, x_max = min(x_coords), max(x_coords)

    # Formula: w2*x + w3*y - eff_threshold = 0  => y = (eff_threshold - w2*x) / w3
    y_min = (eff_threshold - w[2] * x_min) / w[3]
    y_max = (eff_threshold - w[2] * x_max) / w[3]

    plt.plot([x_min, x_max], [y_min, y_max], 'g-', label='Decision Hyperplane')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.legend()
    plt.show()


def main():
    print("Welcome to Petal Classifier")
    p = trainAndTest(False)
    while True:
        print("Here are the options(\\x to leave):")
        print("1. Show Petal Classifier Plot")
        print("2. Test Petal Classifier with your own input")
        val = input()
        if val == "1":
            fourDim()
        elif val == "2":
            uInput = (input("Enter 4 numbers: "))
            nums = [float(x) for x in uInput.split()]
            prediction = p.predict(nums)
            label = "Setosa (1)" if prediction == 1 else "Versicolor (0)"
            print(f"\nResult: {label}\n\n\n")
        elif val == "\\x":
            exit(0)

if __name__ == "__main__":
    main()
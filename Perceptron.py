from EvaluationMetrics import EvaluationMetrics


class Perceptron:
    dimension: int
    weights: list
    threshold: float
    alpha: float
    beta: float

    def __init__(self, dimension: int, threshold: float, alpha: float, beta: float):
        self.dimension = dimension
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        self.weights = [0] * dimension




    def predict(self, input):
        net = 0.0
        for i in range(len(self.weights)):
            net += self.weights[i] * input[i]

        net -= self.threshold
        return 1 if net >= 0 else 0

    def train(self, inputs,labels,epochs,learning_rate = None,showResults = True):
        if learning_rate is None:
            learning_rate = self.alpha
        mistake = True
        epochsCount = 0
        guessed = []
        accuracy_history = []
        while mistake and epochsCount < epochs:
            guessed.clear()
            mistake = False
            for i in range(len(inputs)):
                net = self.predict(inputs[i])
                error = labels[i] - net
                guessed.append(net)
                if error != 0:
                    mistake = True
                    for j in range(len(self.weights)):
                        self.weights[j] = self.weights[j] + error * learning_rate * inputs[i][j]
                    self.threshold -= error * self.beta
            if showResults:
                print(self.weights)
            epochsCount += 1
            accuracy , values = EvaluationMetrics.measureAccuracy(guessed, labels)
            accuracy_history.append(accuracy)
            if showResults:
                print(f"Epoch {epochsCount}: {accuracy:.1f}% ({values})")
            if not mistake:
                break
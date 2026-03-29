class EvaluationMetrics:
    @staticmethod
    def measureAccuracy(realClasses, predictedClasses):
        correct = sum(1 for r, p in zip(realClasses, predictedClasses) if r == p)
        return correct / len(realClasses) * 100 , f"{correct} / {len(realClasses)}"
import numpy as np


class Accuracy:
    def calculate(self, predictions, y):
        comparisions = self.compare(predictions, y)
        accuracy = np.mean(comparisions)
        self.accumulated_sum += np.sum(comparisions)
        self.accumulated_count += len(comparisions)
        return accuracy

    def calculate_accumulated(self):
        return self.accumulated_sum / self.accumulated_count

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def compare(self, predictions, y):
        print("This method must be overridden")
        return -1


class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision


class Accuracy_Categorical(Accuracy):
    def __init__(self, *, binary=False):
        self.binary = binary

    def init(self, y):
        pass

    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
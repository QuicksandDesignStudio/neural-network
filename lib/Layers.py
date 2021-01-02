import numpy as np


class Layer_Dense:
    def __init__(
        self,
        n_inputs,
        n_neurons,
        weight_regulaizer_l1=0,
        weight_regulaizer_l2=0,
        bias_regularizer_l1=0,
        bias_regularizer_l2=0,
    ):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        self.weight_regulaizer_l1 = weight_regulaizer_l1
        self.weight_regulaizer_l2 = weight_regulaizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs, training):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    def backward(self, dvalues):
        self.dweights = np.dot(
            self.inputs.T, dvalues
        )  # derivative with respect to weights is inputs d(input * weight)/d(weight) = input
        self.dbiases = np.sum(
            dvalues, axis=0, keepdims=True
        )  # derivative of sum is 1 here so just sum up dvalues

        # adding the regularization derivatives for weights and biases if needed
        if self.weight_regulaizer_l1 > 0:
            dl1 = np.ones_like(self.weights)
            dl1[self.weights < 0] = -1
            self.dweights += self.weight_regulaizer_l1 * dl1

        if self.weight_regulaizer_l2 > 0:
            self.dweights += 2 * self.weight_regulaizer_l2 * self.weights

        if self.bias_regularizer_l1 > 0:
            dl1 = np.ones_like(self.biases)
            dl1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dl1

        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        self.dinputs = np.dot(
            dvalues, self.weights.T
        )  # derivative with respect to inputs is weights d(input * weight)/d(input) = weight

    def get_parameters(self):
        return self.weights, self.biases

    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases


class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs, training):
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = (
            np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        )
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = (
            dvalues * self.binary_mask
        )  # derivative of the dropout operation is equal to the binary mask - see derivation


class Layer_Input:
    def forward(self, inputs, training):
        self.output = inputs
import numpy as np
import pandas as pd


# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivation of Sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, layers, alpha=0.001, ):
        # Number of layers & nodes
        self.layers = layers
        # Learning rate
        self.alpha = alpha
        # Init W & b
        self.W = []
        self.b = []
        self.init_state()

    def init_state(self):
        for i in range(0, len(self.layers - 1)):
            w = np.random.rand(self.layers[i], self.layers[i + 1])
            b = np.zeros((self.layers[i + 1], 1))
            self.W.append(w / self.layers[i])
            self.b.append(b)

    def fit_partial(self, x, y):
        A = [x]

        # Feed Forward
        previous_state = A[-1]
        for i in range(len(self.layers) - 1):
            A.append(sigmoid(np.dot(previous_state, self.W[i]) + self.b[i].T))

        # Back propagation
        y = y.reshape(-1, 1)
        dA = [-(y / A[-1] - (1 - y) / (1 - A[-1]))]
        dW = []
        db = []
        for i in reversed(range(0, len(self.layers) - 1)):
            dw_ = np.dot((A[i]).T, dA[-1] * sigmoid_derivative(A[i + 1]))
            db_ = (np.sum(dA[-1] * sigmoid_derivative(A[i + 1]), 0)).reshape(-1, 1)
            dA_ = np.dot(dA[-1] * sigmoid_derivative(A[i + 1]), self.W[i].T)
            dW.append(dw_)
            db.append(db_)
            dA.append(dA_)

        # Reverse dW, db
        dW = dW[::-1]
        db = db[::-1]

        # Apply Gradient Descent
        for i in range(len(self.layers) - 1):
            self.W[i] = self.W[i] - self.alpha * dW[i]
            self.b[i] = self.b[i] - self.alpha * db[i]

        # TODO: Add fit, predict, and calculate_loss functions

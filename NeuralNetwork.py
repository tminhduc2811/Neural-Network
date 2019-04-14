import numpy as np
import pandas as pd


# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivation of Sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, layers, alpha=0.001):
        # Number of layers & nodes
        self.layers = layers
        # Learning rate
        self.alpha = alpha
        # Init W & b
        self.W = []
        self.b = []
        self.init_state()

    def init_state(self):
        for i in range(len(self.layers) - 1):
            w = np.random.randn(self.layers[i], self.layers[i + 1])
            # print('w = ', w)
            b = np.zeros((self.layers[i + 1], 1))
            # print('b = ', b)
            self.W.append(w / self.layers[i])
            self.b.append(b)

    def fit_partial(self, x, y):
        A = [x]

        # Feed Forward
        output = A[-1]
        for i in range(len(self.layers) - 1):
            output = sigmoid(np.dot(output, self.W[i]) + self.b[i].T)
            A.append(output)
        # Back propagation
        y = y.reshape(-1, 1)
        dA = [-(y / A[-1] - (1 - y) / (1 - A[-1]))]
        dW = []
        db = []
        for i in reversed(range(len(self.layers) - 1)):
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

    # fit
    def fit(self, x, y, epochs=30, verbose=10):
        for epoch in range(epochs):
            self.fit_partial(x, y)
            if epoch % verbose == 0:
                loss = self.calculate_loss(x, y)
                print('Epoch {}, loss {}'.format(epoch, loss))

    # Predict
    def predict(self, x):
        for i in range(len(self.layers) - 1):
            x = sigmoid(np.dot(x, self.W[i]) + self.b[i].T)
        return x

    # Loss calculation
    def calculate_loss(self, x, y):
        y_predict = self.predict(x)
        return -(np.sum(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict)))

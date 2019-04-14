from NeuralNetwork import NeuralNetwork
import pandas as pd
# Data 1
print('Starting training data 1')
data = pd.read_csv('dataset.csv').values
N, d = data.shape
X = data[:, 0:d - 1].reshape(-1, d - 1)
y = data[:, 2].reshape(-1, 1)
p = NeuralNetwork([X.shape[1], 2, 1], 0.1)
p.fit(X, y, 10000, 100)
# This will print the result = 0
print('Result for x = [5 0.1] ', p.predict([5, 0.1]))
# This will print the result = 1
print('Result for x = [10 0.8] ', p.predict([10, 0.8]))

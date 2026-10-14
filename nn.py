"""Backpropagation neural network (NumPy, sigmoid activations, binary output)."""
import numpy as np

sigmoid = lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500)))


class NeuralNet:
    def __init__(self, layers, lr=0.5, epochs=1500, seed=42):
        rng = np.random.default_rng(seed)
        self.lr, self.epochs = lr, epochs
        self.W = [rng.normal(0, 0.3, (a, b)) for a, b in zip(layers[:-1], layers[1:])]
        self.b = [np.zeros((1, n)) for n in layers[1:]]

    def forward(self, X):
        acts = [X]
        for W, b in zip(self.W, self.b):
            acts.append(sigmoid(acts[-1] @ W + b))
        return acts  # acts[-1] = output, acts[1] = first hidden layer

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        for _ in range(self.epochs):
            acts = self.forward(X)
            delta = acts[-1] - y                                  # output error (cross-entropy + sigmoid)
            for i in reversed(range(len(self.W))):                   # backpropagate
                gW, gb = acts[i].T @ delta / len(X), delta.mean(0, keepdims=True)
                if i: delta = (delta @ self.W[i].T) * acts[i] * (1 - acts[i])
                self.W[i] -= self.lr * gW
                self.b[i] -= self.lr * gb
        return self

    def predict_proba(self, X):
        return self.forward(X)[-1].ravel()

    def predict(self, X, t=0.5):
        return (self.predict_proba(X) >= t).astype(int)

    def hidden(self, X):
        return self.forward(X)[1]

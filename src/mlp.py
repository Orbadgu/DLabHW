import numpy as np


class CustomMLP:

    def __init__(self, input_size, hidden_size, output_size=1, learning_rate=0.01, n_steps=1000, lambd=0.0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.lambd = lambd
        self.parameters = {}
        self.history = {'train_loss': [], 'val_loss': []}

        self._initialize_parameters()

    def _initialize_parameters(self):
        np.random.seed(42)
        self.parameters['W1'] = np.random.randn(self.hidden_size, self.input_size) * 0.1
        self.parameters['b1'] = np.zeros((self.hidden_size, 1))
        self.parameters['W2'] = np.random.randn(self.output_size, self.hidden_size) * 0.1
        self.parameters['b2'] = np.zeros((self.output_size, 1))

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-np.clip(Z, -250, 250)))

    def _forward_propagation(self, X):
        W1, b1 = self.parameters['W1'], self.parameters['b1']
        W2, b2 = self.parameters['W2'], self.parameters['b2']

        Z1 = np.dot(W1, X.T) + b1
        A1 = self._sigmoid(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self._sigmoid(Z2)

        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    def _compute_cost(self, A2, Y):
        m = Y.shape[0]
        W1, W2 = self.parameters['W1'], self.parameters['W2']

        cross_entropy_cost = - (1 / m) * np.sum(Y.T * np.log(A2 + 1e-8) + (1 - Y.T) * np.log(1 - A2 + 1e-8))

        L2_regularization_cost = (self.lambd / (2 * m)) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

        cost = cross_entropy_cost + L2_regularization_cost
        return np.squeeze(cost)

    def _backpropagation(self, X, Y, cache):
        m = X.shape[0]
        A1, A2 = cache['A1'], cache['A2']
        W1, W2 = self.parameters['W1'], self.parameters['W2']

        dZ2 = A2 - Y.T

        dW2 = (1 / m) * np.dot(dZ2, A1.T) + (self.lambd / m) * W2
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.dot(W2.T, dZ2) * (A1 * (1 - A1))

        dW1 = (1 / m) * np.dot(dZ1, X) + (self.lambd / m) * W1
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return grads

    def _update_parameters(self, grads):
        self.parameters['W1'] -= self.learning_rate * grads['dW1']
        self.parameters['b1'] -= self.learning_rate * grads['db1']
        self.parameters['W2'] -= self.learning_rate * grads['dW2']
        self.parameters['b2'] -= self.learning_rate * grads['db2']

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        for i in range(self.n_steps):
            A2, cache = self._forward_propagation(X_train)
            cost = self._compute_cost(A2, y_train)
            self.history['train_loss'].append(cost)

            grads = self._backpropagation(X_train, y_train, cache)
            self._update_parameters(grads)

            if X_val is not None and y_val is not None:
                A2_val, _ = self._forward_propagation(X_val)
                val_cost = self._compute_cost(A2_val, y_val)
                self.history['val_loss'].append(val_cost)

            if verbose and i % (self.n_steps // 10) == 0:
                val_text = f" | Val Cost: {val_cost:.4f}" if X_val is not None else ""
                print(f"Step {i} | Train Cost: {cost:.4f}{val_text}")

    def predict(self, X):
        A2, _ = self._forward_propagation(X)
        predictions = (A2 > 0.5).astype(int)
        return predictions.T
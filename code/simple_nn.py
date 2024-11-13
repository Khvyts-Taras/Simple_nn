import numpy as np


def get_batches(x, y, batch_size):
    for i in range(0, len(x), batch_size):
        yield x[i:i + batch_size], y[i:i + batch_size]

def MAELoss(target, res):
	loss = np.mean(np.abs(target - res))
	target_grad = np.sign(target - res)
	res_grad = np.sign(res - target)
	return loss, res_grad, target_grad

def MSELoss(target, res):
	loss = np.mean((target - res)**2)
	target_grad = 2 * (target - res)
	res_grad = 2 * (res - target)
	return loss, res_grad, target_grad


class Sigmoid:
	def forward(self, x):
		self.y = 1/(1 + np.exp(-x))
		return self.y

	def backward(self, grad):
		df = self.y * (1 - self.y)
		return df*grad


class LeakyReLU:
	def forward(self, x, alpha=0.01):
		self.x = x
		self.alpha = alpha
		return np.where(x > 0, x, x * alpha)

	def backward(self, grad):
		return np.where(self.x > 0, 1, self.alpha) * grad


class Linear:
    def __init__(self, n_inp, n_out):
        n = 6**0.5 / (n_inp+n_out)**0.5
        self.w = np.random.uniform(-n, n, (n_inp, n_out))
        n = 6**0.5 / (1+n_out)**0.5
        self.b = np.random.uniform(-n, n, (1, n_out))

    def forward(self, x):
        self.x = x
        return self.x @ self.w + self.b

    def backward(self, grad):
        self.gw = self.x.T @ grad
        self.gb = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.w.T

    def step(self, lr, norm_value=1):
        grad_norm = np.linalg.norm(self.gw)
        if grad_norm > norm_value:
            self.gw = (self.gw / grad_norm) * norm_value
        grad_norm = np.linalg.norm(self.gb)
        if grad_norm > norm_value:
            self.gb = (self.gb / grad_norm) * norm_value
        self.w -= self.gw * lr
        self.b -= self.gb * lr

    def save_weights(self, file_path):
        np.savez(file_path, w=self.w, b=self.b)

    def load_weights(self, file_path):
        data = np.load(file_path)
        self.w = data['w']
        self.b = data['b']


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, err):
        for layer in reversed(self.layers):
            err = layer.backward(err)
        return err

    def step(self, learning_rate):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.step(learning_rate)

    def save_weights(self, file_path):
        weights = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                weights[f"w_{i}"] = layer.w
                weights[f"b_{i}"] = layer.b
        np.savez(file_path, **weights)

    def load_weights(self, file_path):
        data = np.load(file_path)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                layer.w = data[f"w_{i}"]
                layer.b = data[f"b_{i}"]
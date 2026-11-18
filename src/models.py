import numpy as np

def activate(z, name):
    if name == "relu": return np.maximum(0, z)
    if name == "sigmoid": return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
    if name == "tanh": return np.tanh(z)
    if name == "linear": return z
    raise ValueError(f"Unknown activation: {name}")

def activation_grad(z, name):
    if name == "relu": return (z > 0).astype(np.float32)
    if name == "sigmoid":
        s = activate(z, name); return s * (1 - s)
    if name == "tanh":
        t = np.tanh(z); return 1 - t * t
    if name == "linear": return np.ones_like(z)
    raise ValueError(f"Unknown activation: {name}")

class NumpyNetwork:
    """Explicit gradient descent/backprop implementation using NumPy."""
    def __init__(self, n_features, model="linear", hidden=16, activation="tanh", seed=42):
        rng = np.random.default_rng(seed)
        self.model, self.activation = model, activation
        if model == "linear":
            self.W = rng.normal(0, 0.1, (n_features, 1)).astype(np.float32)
            self.b = np.zeros((1,), dtype=np.float32)
        elif model == "mlp":
            self.W1 = rng.normal(0, np.sqrt(2/n_features), (n_features, hidden)).astype(np.float32)
            self.b1 = np.zeros((1, hidden), dtype=np.float32)
            self.W2 = rng.normal(0, np.sqrt(2/hidden), (hidden, 1)).astype(np.float32)
            self.b2 = np.zeros((1,), dtype=np.float32)
        else:
            raise ValueError("model must be 'linear' or 'mlp'")

    def predict(self, X):
        if self.model == "linear":
            return X @ self.W + self.b
        self.z1 = X @ self.W1 + self.b1
        self.a1 = activate(self.z1, self.activation)
        return self.a1 @ self.W2 + self.b2

    def train_step(self, X, y, lr):
        n = len(X)
        pred = self.predict(X)
        loss = np.mean((pred-y)**2)
        # d(MSE)/d(prediction)
        d_out = (2.0/n) * (pred-y)
        if self.model == "linear":
            dW = X.T @ d_out
            db = np.sum(d_out, axis=0)
            self.W -= lr*dW
            self.b -= lr*db
        else:
            dW2 = self.a1.T @ d_out
            db2 = np.sum(d_out, axis=0)
            da1 = d_out @ self.W2.T
            dz1 = da1 * activation_grad(self.z1, self.activation)
            dW1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0)
            self.W2 -= lr*dW2; self.b2 -= lr*db2
            self.W1 -= lr*dW1; self.b1 -= lr*db1
        return float(loss)

    def fit(self, X, y, epochs, lr):
        history=[]
        for _ in range(epochs):
            history.append(self.train_step(X, y, lr))
        return history

    def state(self):
        if self.model == "linear":
            return {"model": self.model, "W": self.W.tolist(), "b": self.b.tolist()}
        return {"model": self.model, "activation": self.activation,
                "W1": self.W1.tolist(), "b1": self.b1.tolist(),
                "W2": self.W2.tolist(), "b2": self.b2.tolist()}

def train_autograd(backend, X, y, model, hidden, activation, epochs, lr, seed=42):
    if backend == "pytorch":
        import torch
        torch.manual_seed(seed)
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                if model == "linear":
                    self.layers = torch.nn.Linear(X.shape[1], 1)
                else:
                    acts = {"relu": torch.nn.ReLU, "sigmoid": torch.nn.Sigmoid,
                            "tanh": torch.nn.Tanh, "linear": torch.nn.Identity}
                    self.layers = torch.nn.Sequential(
                        torch.nn.Linear(X.shape[1], hidden), acts[activation](),
                        torch.nn.Linear(hidden, 1))
            def forward(self, x): return self.layers(x)
        net=Net()
        opt=torch.optim.SGD(net.parameters(), lr=lr)
        loss_fn=torch.nn.MSELoss()
        xt=torch.tensor(X, dtype=torch.float32); yt=torch.tensor(y, dtype=torch.float32)
        history=[]
        for _ in range(epochs):
            opt.zero_grad(); pred=net(xt); loss=loss_fn(pred, yt)
            loss.backward(); opt.step(); history.append(float(loss.detach()))
        return lambda xx: net(torch.tensor(xx, dtype=torch.float32)).detach().numpy(), history
    if backend == "tensorflow":
        import tensorflow as tf
        tf.random.set_seed(seed)
        if model == "linear":
            net=tf.keras.Sequential([tf.keras.layers.Input((X.shape[1],)), tf.keras.layers.Dense(1)])
        else:
            net=tf.keras.Sequential([
                tf.keras.layers.Input((X.shape[1],)),
                tf.keras.layers.Dense(hidden, activation=activation),
                tf.keras.layers.Dense(1)])
        net.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr), loss="mse")
        hist=net.fit(X, y, epochs=epochs, verbose=0)
        return lambda xx: net.predict(xx, verbose=0), [float(v) for v in hist.history["loss"]]
    raise ValueError("backend must be numpy, tensorflow, or pytorch")

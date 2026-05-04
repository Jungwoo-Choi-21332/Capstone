import numpy as np


# ---------------------------
# Activation functions
# ---------------------------
def tanh(x):
    """Hyperbolic tangent activation"""
    return np.tanh(x)


def relu(x):
    """ReLU activation"""
    return np.maximum(0, x)


# ---------------------------
# Spectral radius scaling
# ---------------------------
def compute_spectral_radius(W):
    """Compute spectral radius (largest eigenvalue magnitude)"""
    eigenvalues = np.linalg.eigvals(W)
    return np.max(np.abs(eigenvalues))


def scale_spectral_radius(W, desired_radius=0.9):
    """
    Scale matrix W to have a desired spectral radius
    """
    radius = compute_spectral_radius(W)

    if radius == 0:
        return W

    return W * (desired_radius / radius)


# ---------------------------
# Weight initialization
# ---------------------------
def initialize_reservoir(size, sparsity=0.1, scale=1.0, seed=None):
    """
    Initialize sparse reservoir weight matrix
    """
    rng = np.random.default_rng(seed)

    W = rng.uniform(-1, 1, (size, size))

    # Apply sparsity mask
    mask = rng.random((size, size)) < sparsity
    W = W * mask

    return W * scale


def initialize_input_weights(input_dim, reservoir_size, scale=1.0, seed=None):
    """
    Initialize input weight matrix
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, (reservoir_size, input_dim)) * scale


# ---------------------------
# State update (core of ESN)
# ---------------------------
def update_state(x_prev, u, W, Win, activation=tanh, leaking_rate=1.0):
    """
    ESN state update equation

    x(t) = (1 - alpha)*x(t-1) + alpha * f(Wx(t-1) + Win*u(t))
    """
    pre_activation = np.dot(W, x_prev) + np.dot(Win, u)
    x_new = activation(pre_activation)

    # Leaky integration
    return (1 - leaking_rate) * x_prev + leaking_rate * x_new


# ---------------------------
# Readout training (ridge)
# ---------------------------
def ridge_regression(X, Y, reg=1e-6):
    """
    X: (samples, features)
    Y: (samples, output_dim)
    """

    I = np.eye(X.shape[1])

    W_out = np.linalg.inv(X.T @ X + reg * I) @ X.T @ Y

    return W_out.T


# ---------------------------
# Prediction
# ---------------------------
def predict(W_out, X):
    """Linear readout prediction"""
    return np.dot(X, W_out.T)
import numpy as np

from math_function import (
    initialize_reservoir,
    initialize_input_weights,
    scale_spectral_radius,
    update_state,
    ridge_regression,
    tanh
)


class ESN:
    def __init__(
        self,
        input_dim,
        reservoir_size=200,
        spectral_radius=0.9,
        sparsity=0.1,
        input_scaling=1.0,
        leaking_rate=1.0,
        reg=1e-6,
        seed=None
    ):
        """
        Echo State Network (ESN)
        """

        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.leaking_rate = leaking_rate
        self.reg = reg

        # Initialize weights
        self.W = initialize_reservoir(
            reservoir_size, sparsity=sparsity, seed=seed
        )
        self.W = scale_spectral_radius(self.W, spectral_radius)

        self.Win = initialize_input_weights(
            input_dim, reservoir_size, scale=input_scaling, seed=seed
        )

        self.Wout = None  # trained later

    # ---------------------------
    # Reservoir forward pass
    # ---------------------------
    def _compute_states(self, X_seq):
        """
        X_seq: (samples, time_steps, input_dim)
        return: states (samples, reservoir_size)
        """

        all_states = []

        for seq in X_seq:
            x = np.zeros(self.reservoir_size)

            for u in seq:
                x = update_state(
                    x,
                    u,
                    self.W,
                    self.Win,
                    activation=tanh,
                    leaking_rate=self.leaking_rate
                )

            all_states.append(x)

        return np.array(all_states)

    # ---------------------------
    # Training
    # ---------------------------
    def fit(self, X_seq, y):
        """
        Train readout layer
        """

        # Compute reservoir states
        states = self._compute_states(X_seq)

        # Add bias term
        states_aug = np.hstack([states, np.ones((states.shape[0], 1))])

        # Train readout (ridge regression)
        self.Wout = ridge_regression(
            states_aug,
            y.reshape(-1, 1),
            reg=self.reg
        )

    # ---------------------------
    # Prediction
    # ---------------------------
    def predict(self, X_seq):
        """
        Predict output
        """

        states = self._compute_states(X_seq)

        # Add bias
        states_aug = np.hstack([states, np.ones((states.shape[0], 1))])

        y_pred = np.dot(states_aug, self.Wout.T)

        return y_pred.flatten()

    # ---------------------------
    # Classification output
    # ---------------------------
    def predict_class(self, X_seq, threshold=0.5):
        """
        Convert to binary class
        """

        y_pred = self.predict(X_seq)

        return (y_pred > threshold).astype(int)
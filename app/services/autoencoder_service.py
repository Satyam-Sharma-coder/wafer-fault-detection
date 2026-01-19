import numpy as np

class AutoencoderService:
    def __init__(self, autoencoder_model):
        self.autoencoder = autoencoder_model

    def compute_errors(self, X):
        X_hat = self.autoencoder.reconstruct(X)
        errors = np.mean((X - X_hat) ** 2, axis=1)
        return errors

    def find_threshold(self, errors):
        return errors.mean() + 3 * errors.std()

    def detect_faults(self, errors, threshold):
        return errors > threshold

import numpy as np
from app.services.autoencoder_service import AutoencoderService

def test_autoencoder_train_and_error():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(128, 10)).astype(np.float32)
    ae = AutoencoderService(input_dim=10, hidden_dim=8, epochs=1, lr=1e-3, device="cpu")
    ae.train(X)
    errs = ae.reconstruction_error(X)
    assert errs.shape == (128,)
    assert np.isfinite(errs).all()
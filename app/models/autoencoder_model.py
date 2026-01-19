from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

class AutoencoderModel:
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model = self._build()

    def _build(self):
        inp = Input(shape=(self.input_dim,))
        x = Dense(128, activation="relu")(inp)
        x = Dense(64, activation="relu")(x)
        x = Dense(128, activation="relu")(x)
        out = Dense(self.input_dim, activation="sigmoid")(x)

        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer=Adam(0.001), loss="mse")
        return model

    def train(self, X_train, epochs=50, batch_size=32):
        self.model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size)

    def reconstruct(self, X):
        return self.model.predict(X)

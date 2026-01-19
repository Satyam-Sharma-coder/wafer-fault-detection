from tensorflow.keras.models import load_model
import joblib

class AutoencoderPersistence:
    @staticmethod
    def save(model, path):
        model.model.save(path)

    @staticmethod
    def load(path):
        return load_model(path)

class ClassifierPersistence:
    @staticmethod
    def save(model, path):
        joblib.dump(model.model, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
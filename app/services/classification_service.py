class ClassificationService:
    def __init__(self, classifier):
        self.classifier = classifier

    def train(self, X, y):
        self.classifier.train(X, y)

    def predict(self, X):
        return self.classifier.predict(X)
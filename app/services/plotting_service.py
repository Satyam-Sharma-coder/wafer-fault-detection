import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score

class PlottingService:
    def plot_errors(self, errors, threshold, output_path):
        plt.figure()
        plt.hist(errors, bins=50)
        plt.axvline(threshold, linestyle="--")
        plt.title("Reconstruction Error Distribution")
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        plt.savefig(output_path)
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, output_path):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.title("Confusion Matrix")
        plt.savefig(output_path)
        plt.close()

    def compute_metrics(self, y_true, y_pred):
        precision = precision_score(y_true, y_pred, pos_label=1)
        recall = recall_score(y_true, y_pred, pos_label=1)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        return precision, recall, f1

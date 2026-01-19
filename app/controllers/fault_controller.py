from flask import Blueprint, render_template
from app.repositories.dataset_repository import DatasetRepository
from app.services.preprocessing_service import PreprocessingService
from app.models.autoencoder_model import AutoencoderModel
from app.services.autoencoder_service import AutoencoderService
from app.services.plotting_service import PlottingService
from app.services.severity_service import SeverityService
from app.models.classifiers import RandomForestModel
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import os
import numpy as np
from tensorflow.keras.optimizers import Adam

fault_bp = Blueprint("fault", __name__)

@fault_bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@fault_bp.route("/run", methods=["POST"])
def run_pipeline():
    repo = DatasetRepository("data/uci-secom.csv")
    df = repo.load()

    pre = PreprocessingService()
    X, y = pre.preprocess(df)

    healthy = X[y == -1]

    ae_path = "saved_models/autoencoder.h5"

    if os.path.exists(ae_path):
        ae_model = load_model(ae_path, compile=False)

        autoencoder = AutoencoderModel(X.shape[1])
        autoencoder.model = ae_model

        # Recompile manually (avoids 'mse' error)
        autoencoder.model.compile(
        optimizer=Adam(0.001),
        loss="mse"
        )
    else:
        autoencoder = AutoencoderModel(X.shape[1])
        autoencoder.train(healthy)
        autoencoder.model.save(ae_path)

    ae_service = AutoencoderService(autoencoder)
    errors = ae_service.compute_errors(X)
    threshold = ae_service.find_threshold(errors)
    detected_faults = ae_service.detect_faults(errors, threshold)

    # Map unsupervised output → binary labels
    y_pred_unsupervised = np.where(detected_faults, 1, -1)

    # Metrics
    precision = precision_score(y, y_pred_unsupervised, pos_label=1)
    recall = recall_score(y, y_pred_unsupervised, pos_label=1)
    f1 = f1_score(y, y_pred_unsupervised, pos_label=1)

    plotter = PlottingService()
    error_plot = "app/static/plots/errors.png"
    cm_plot = "app/static/plots/confusion_matrix.png"

    plotter.plot_errors(errors, threshold, error_plot)
    plotter.plot_confusion_matrix(y, y_pred_unsupervised, cm_plot)

    severity_service = SeverityService()
    severity_labels = severity_service.assign_severity(errors, threshold)

    severity_counts = {
        "Normal": severity_labels.count("Normal"),
        "Low": severity_labels.count("Low"),
        "Medium": severity_labels.count("Medium"),
        "High": severity_labels.count("High")
    }

    return render_template(
        "result.html",
        total=len(X),
        faulty=int(detected_faults.sum()),
        threshold=round(float(threshold), 6),
        precision=round(float(precision), 4),
        recall=round(float(recall), 4),
        f1=round(float(f1), 4),
        severity=severity_counts,
        error_plot="/static/plots/errors.png",
        cm_plot="/static/plots/confusion_matrix.png"
    )

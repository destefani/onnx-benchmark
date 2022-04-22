from fileinput import filename
from pathlib import Path
from joblib import dump, load
import time
from matplotlib.font_manager import _Weight
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression


# Logistic regression report

config = {
    "n_features": [10, 100, 1000, 10000],
    "n_samples": [1e4, 1e5, 1e6, 1e7],
    "model_type": "logistic_regression",
}


def train_data(n_features):
    """Generate training data"""
    X = np.random.rand(1, n_features)
    y = np.random.randint(0, size=1)
    return X, y


def test_data(n_features, n_samples):
    """Generate test data"""
    return np.random.rand(n_samples, n_features)


def fit_model(X, y, model=None):
    """Fits a model"""
    clf = LogisticRegression()
    clf.fit(X, y)
    return clf


def save_skl(clf, save_directory):
    """Save the sklearn model"""
    filename = Path(save_directory) / "log_reg_model.joblib"
    dump(clf, filename)
    return filename


def save_onnx(clf, n_features, save_directory):
    """Convert the sklearn model to onnx and save it"""
    filename = Path(save_directory) / "log_reg_model.onnx"
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onx = convert_sklearn(clf, initial_types=initial_type)
    with open(filename, "wb") as f:
        f.write(onx.SerializeToString())
    return filename


def skl_report(n_features, n_samples, model_path):
    """Calculates the inference time and model weights of a sklearn model"""
    clf = load(model_path)
    X = test_data(n_features, n_samples)
    inference_time = execution_time(clf.predict(X))
    weights_size = model_path.stat().st_size
    return inference_time, weights_size


def onnx_report(n_features, n_samples, model_path):
    """Calculates the inference time and model weights of an onnx model"""
    sess = rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    X = test_data(n_features, n_samples)
    X_test_onnx = X.astype(np.float32)
    inference_time = execution_time(sess.run([label_name], {input_name: X_test_onnx}))
    weights_size = model_path.stat().st_size
    return inference_time, weights_size


def execution_time(process):
    """Calculates the execution time of a function"""
    start_time = time.time()
    (lambda: process)()
    end_time = time.time()
    return end_time - start_time

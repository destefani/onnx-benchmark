from pathlib import Path
from joblib import dump, load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression


# Logistic regression report

config = {
    'n_features': [10, 100, 1000, 10000],
    'n_samples': [1e4, 1e5, 1e6, 1e7],
    'model_type': 'logistic_regression'
}


def generate_test_data(n_features, n_samples):
    return np.random.randn(n_samples, n_features)

def fit_model(X, y):
    clf = LogisticRegression()
    clf.fit(X, y)
    return clf

def save_skl_model(clf, path):
    dump(clf, path)
    

def convert2onnx(clf, n_features, path):
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    onx = convert_sklearn(clf, initial_types=initial_type)
    with open("models/log_reg_model.onnx", "wb") as f:
        f.write(onx.SerializeToString())

def create_report(cfg):
    df = pd.DataFrame(columns=[
        'model_type',
        'n_features',
        'n_samples',
        'file_size',
        'inference_time'
        ])
    pass





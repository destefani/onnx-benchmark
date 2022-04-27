from fileinput import filename
from pathlib import Path
from joblib import dump, load
import time
import click
import numpy as np
import pandas as pd
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# ----------------------------------------------------------------------------


def check_directory(path):
    Path(path).mkdir(exist_ok=True)


def train_data(n_features):
    """Generate training data"""
    X = np.random.rand(10, n_features)
    y = np.random.randint(2, size=10)
    return X, y


def test_data(n_features, n_samples) -> np.ndarray:
    """Generate test data"""
    return np.random.rand(int(n_samples), n_features)


def fit_model(X, y, model="logistic_regression"):
    """Fits a model"""
    if model == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
        clf.fit(X, y)
    if model == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
    
    if model == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier()
        clf.fit(X, y)
    if model == "svm":
        from sklearn.svm import SVC
        clf = SVC()
        clf.fit(X, y)
    return clf


def save_skl(clf, model, n_features, save_directory) -> Path:
    """Save the sklearn model"""
    filename = Path(save_directory) / f"{model}_{n_features}.joblib"
    dump(clf, filename)
    return filename


def save_onnx(clf, model, n_features, save_directory) -> Path:
    """Convert the sklearn model to onnx and save it"""
    filename = Path(save_directory) / f"{model}_{n_features}.onnx"
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onx = convert_sklearn(clf, initial_types=initial_type)
    with open(filename, "wb") as f:
        f.write(onx.SerializeToString())
    return filename


def skl_stats(n_features, n_samples, model_path) -> tuple:
    """Calculates the inference time and model weights of a sklearn model"""
    clf = load(model_path)
    X = test_data(n_features, n_samples)
    inference_time = execution_time(clf.predict(X))
    weights_size = model_path.stat().st_size / 1024  # in KB
    return inference_time, weights_size


def onnx_stats(n_features, n_samples, model_path) -> tuple:
    """Calculates the inference time and model weights of an onnx model"""
    sess = rt.InferenceSession(str(model_path))
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    X = test_data(n_features, n_samples)
    X_test_onnx = X.astype(np.float32)
    inference_time = execution_time(sess.run([label_name], {input_name: X_test_onnx}))
    weights_size = model_path.stat().st_size / 1024  # in KB
    return inference_time, weights_size


def execution_time(process) -> float:
    """Calculates the execution time of a function in milliseconds"""
    process = lambda: process
    start_time = time.time()
    process()
    end_time = time.time()
    return (end_time - start_time) * 1000


def init_report_df():
    """Initialize the report dataframe"""
    return pd.DataFrame(
        columns=[
            "model",
            "n_features",
            "n_samples",
            "skl_inference_time_ms",
            "skl_weights_size_KB",
            "onnx_inference_time_ms",
            "onnx_weights_size_KB",
        ]
    )


def model_report(
    n_features: list, n_samples: list, model: str, save_directory="results"
) -> pd.DataFrame:
    """Calculates the inference time and model weights of a model with different parameters"""

    check_directory(save_directory)  # add temporary directory

    report_df = init_report_df()

    for features in n_features:
        # Define model
        X_train, y_train = train_data(features)
        clf = fit_model(X_train, y_train, model)
        
        for samples in n_samples:
            print(f"{model} with {features} features and {samples} samples")

            # Run models
            skl_model = save_skl(clf, model, features, save_directory)
            onnx_model = save_onnx(clf, model, features, save_directory)

            # Calculate inference time and model weights
            skl_inference_time, skl_weights_size = skl_stats(
                features, samples, skl_model
            )
            onnx_inference_time, onnx_weights_size = onnx_stats(
                features, samples, onnx_model
            )

            # Save results
            results = {
                "model": model,
                "n_features": features,
                "n_samples": samples,
                "skl_inference_time_ms": skl_inference_time,
                "skl_weights_size_KB": skl_weights_size,
                "onnx_inference_time_ms": onnx_inference_time,
                "onnx_weights_size_KB": onnx_weights_size,
            }
            results_df = pd.DataFrame(results, index=[0])
            report_df = pd.concat([report_df, results_df], ignore_index=True)

    return report_df


# ----------------------------------------------------------------------------


# @click.command()
# @click.option(
#     "--outdir", help="Where to save the results", metavar="DIR", default="results/"
# )
# @click.option(
#     "--n_features",
#     help="The number of features in the training data",
#     type=click.IntRange(min=1),
#     default=[10, 100, 1000, 10000],
# )
# @click.option(
#     "--n_samples",
#     help="The number of samples in the test data",
#     metavar="INT",
#     default=[1],
# )

# @click.option(
#     "--models",
#     help="The models to run",
#     type=click.Choice(["logistic_regression", "decision_tree", "random_forest", "svm"]),
#     default=["logistic_regression", "decision_tree", "random_forest", "svm"],
# )

# def main(**kwargs):

#     report_df = pd.DataFrame()
    
#     for model in kwargs["models"]:


#         report = model_report(
#             kwargs["n_features"], kwargs["n_samples"], model, kwargs["outdir"]
#         )
        
    


# ----------------------------------------------------------------------------

if __name__ == "__main__":

    n_features = [10, 100, 1000, 10000]
    # n_samples = [1, 1e2, 1e3, 1e4, 1e5, 1e6]
    n_samples = [1]
    # model = "logistic_regression"
    # model = "random_forest"
    # model = "svm"
    model = "decision_tree"

    # models = ["logistic_regression", "random_forest", "svm", "decision_tree"]1

    results_df = model_report(n_features, n_samples, model)
    results_df.to_csv("results/results.csv", index=False)
    print(results_df)

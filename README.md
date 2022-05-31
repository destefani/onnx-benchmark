# ONNX Benchmarking

## Description
Benchmark footprint and performance of ONNX models.

## Scikit-learn vs ONNX
As default the script will test and compare the inference time and weights size of the default scikit-learn implementations of:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Classifier

All results will be exported as a csv file to the specified directory.


## Basic usage
```
python onnx_benchmark.py
```

## CLI
You can also specify different parameters to test the models. The default parameters are:
```
python onnx_benchmark.py \
    --outdir results/ \
    --n_features 10 --n_features 100 --n_features 1000 --n_features 10000 \
    --n_samples 1 \
    --models logistic_regression --models decision_tree --models random_forest --models svm
```

## Example report
<img src="./assets/report.png">

## To-do:
- Collect information about the system and add it to the report.
- Benchmark runtime footprint.
- Implement more models.

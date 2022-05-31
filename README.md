# ONNX Benchamrking

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

```
                  model n_features n_samples skl_inference_time_ms skl_weights_size_KB onnx_inference_time_ms onnx_weights_size_KB
0   logistic_regression       1000         1              0.000477            8.577148               0.000715            10.382812
1   logistic_regression      10000         1              0.000238           78.889648               0.000238            98.276367
2   logistic_regression     100000         1              0.000477          782.018555               0.000715           977.183594
3   logistic_regression    1000000         1              0.000477         7813.268555               0.000715          9766.249023
4         decision_tree       1000         1              0.000954            1.339844               0.000238             0.858398
5         decision_tree      10000         1              0.000238            1.339844               0.000477             0.858398
6         decision_tree     100000         1              0.000477            1.345703               0.000477             0.860352
7         decision_tree    1000000         1              0.000238            1.345703               0.000715             0.860352
8         random_forest       1000         1              0.000238           56.393555               0.000238            12.357422
9         random_forest      10000         1              0.000477            55.12793               0.000238            11.723633
10        random_forest     100000         1              0.000477           55.758789               0.000715            11.882812
11        random_forest    1000000         1              0.000238           55.618164               0.001192            11.821289
12                  svm       1000         1              0.000238           79.614258               0.000238            49.349609
13                  svm      10000         1              0.000477          782.739258               0.000238           488.802734
14                  svm     100000         1              0.000477         7813.995117               0.000477          4883.337891
15                  svm    1000000         1              0.000715        78126.495117               0.001192         48828.650391
```

## To-do:
- Collect information about the system and add it to the report.
- Benchmark runtime footprint.
- Implement more models.
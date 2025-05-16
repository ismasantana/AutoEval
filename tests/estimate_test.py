import pandas as pd
import numpy as np
import os

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

from evaluators.confidence import ConfidenceThresholdEvaluator

def load_dataset(path):
    df = pd.read_csv(path)
    y = df["Alzheimer"]
    X = df.drop(columns=["Alzheimer"])
    return X, y

def test_estimate_cross_dataset(train_file, test_file):
    train_path = os.path.join("datasets", train_file)
    test_path = os.path.join("datasets", test_file)

    assert os.path.exists(train_path), f"Train dataset not found: {train_path}"
    assert os.path.exists(test_path), f"Test dataset not found: {test_path}"

    X_train, y_train = load_dataset(train_path)
    X_test, y_test = load_dataset(test_path)

    model = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    threshold = 0.7
    evaluator = ConfidenceThresholdEvaluator(
        estimator=model,
        scorer={"acc": accuracy_score},
        threshold=threshold
    )

    evaluator.fit(X_train, y_train)
    estimates = evaluator.estimate(X_test)

    print(f"Estimate shape from model trained on '{train_file}' and tested on '{test_file}':", estimates.shape)
    print("First few estimated probabilities:\n", estimates[:5])
    print("\n")

test_estimate_cross_dataset('geriatria-controle-alzheimerLabel.csv', 'neurologia-controle-alzheimerLabel.csv')
test_estimate_cross_dataset('neurologia-controle-alzheimerLabel.csv', 'geriatria-controle-alzheimerLabel.csv')

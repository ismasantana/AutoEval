# from sklearn.pipeline import make_pipeline
# from sklearn.linear_model import LogisticRegression
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
# import pandas as pd

# from evaluators.confidence import ConfidenceThresholdEvaluator
# from metrics.comparison import compare_real_estimated_cross_accuracy

# def load_dataset(path):
#     df = pd.read_csv(path)
#     X = df.drop(columns=["Alzheimer"])
#     y = df["Alzheimer"]
#     return X, y

# X_geri, y_geri = load_dataset("datasets/geriatria-controle-alzheimerLabel.csv")
# X_neuro, y_neuro = load_dataset("datasets/neurologia-controle-alzheimerLabel.csv")

# model = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=1000))
# evaluator = ConfidenceThresholdEvaluator(model, scorer={"acc": accuracy_score}, threshold=0.7)

# print(">>> Geriatria -> Neurologia")
# compare_real_estimated_cross_accuracy(X_geri, y_geri, X_neuro, y_neuro, evaluator)

# print(">>> Neurologia -> Geriatria")
# compare_real_estimated_cross_accuracy(X_neuro, y_neuro, X_geri, y_geri, evaluator)

import pytest
import sys
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from metrics.comparison import score_error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_score_error_with_callable():
    real = {"acc": 0.8, "f1": 0.75}
    estimated = {"acc": 0.6, "f1": 0.7}

    result = score_error(real, estimated, comparator=mean_absolute_error)

    assert isinstance(result, dict)
    assert "acc" in result and "f1" in result
    assert round(result["acc"], 2) == 0.2
    assert round(result["f1"], 2) == 0.05

def test_score_error_with_dict_comparators():
    real = {"acc": 0.8, "f1": 0.75}
    estimated = {"acc": 0.6, "f1": 0.7}

    comparators = {
        "acc": mean_squared_error,
        "f1": mean_absolute_error
    }

    result = score_error(real, estimated, comparator=comparators)

    assert isinstance(result, dict)
    assert "acc" in result and "f1" in result
    assert round(result["acc"], 2) == 0.04
    assert round(result["f1"], 2) == 0.05  
    
def test_score_error_invalid_comparator():
    real = {"acc": 0.9}
    estimated = {"acc": 0.8}

    with pytest.raises(ValueError):
        score_error(real, estimated, comparator="invalid")

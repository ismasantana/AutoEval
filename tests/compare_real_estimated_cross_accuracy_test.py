from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

from evaluators.confidence import ConfidenceThresholdEvaluator
from metrics.comparison import compare_real_estimated_cross_accuracy

def load_dataset(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["Alzheimer"])
    y = df["Alzheimer"]
    return X, y

X_geri, y_geri = load_dataset("datasets/geriatria-controle-alzheimerLabel.csv")
X_neuro, y_neuro = load_dataset("datasets/neurologia-controle-alzheimerLabel.csv")

model = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=1000))
evaluator = ConfidenceThresholdEvaluator(model, scorer={"acc": accuracy_score}, threshold=0.7)

print(">>> Geriatria -> Neurologia")
compare_real_estimated_cross_accuracy(X_geri, y_geri, X_neuro, y_neuro, evaluator)

print(">>> Neurologia -> Geriatria")
compare_real_estimated_cross_accuracy(X_neuro, y_neuro, X_geri, y_geri, evaluator)

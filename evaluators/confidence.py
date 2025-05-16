from sklearn.base import BaseEstimator
import numpy as np
from collections.abc import Callable

class ConfidenceThresholdEvaluator(BaseEstimator):
    """
        Confidence-based evaluator for classification models.

        Parameters:
        -----------
        estimator : object
            Any model with fit/predict/predict_proba or decision_function methods.
        scorer : callable or dict of str -> callable
            Evaluation function or a dict of multiple evaluation functions.
        threshold : float
            Minimum confidence required to include a prediction.
        limit_to_top_class : bool
            If True, uses only the top class probability as confidence.
        """
    
    def __init__(self, estimator, scorer, threshold=0.8, limit_to_top_class=True):
        self.estimator = estimator
        self.scorer = scorer
        self.threshold = threshold
        self.limit_to_top_class = limit_to_top_class

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def score(self, X, y):
        if hasattr(self.estimator, "predict_proba"):
            probas = self.estimator.predict_proba(X)
            conf = np.max(probas, axis=1) if self.limit_to_top_class else probas
        elif hasattr(self.estimator, "decision_function"):
            decision = self.estimator.decision_function(X)
            conf = np.max(decision, axis=1) if decision.ndim > 1 else np.abs(decision)
        else:
            raise ValueError("The estimator must implement predict_proba or decision_function.")
        
        mask = conf >= self.threshold
        if not np.any(mask):
            return {name: 0.0 for name in self._get_scorer_names()}

        y_pred = self.estimator.predict(X[mask])
        y_true = y[mask]

        if isinstance(self.scorer, dict):
            return {
                name: func(y_true, y_pred)
                for name, func in self.scorer.items()
            }
        elif isinstance(self.scorer, Callable):
            return {'score': self.scorer(y_true, y_pred)}
        else:
            raise ValueError("'scorer' must be a callable or a dict of callables.")

    def _get_scorer_names(self):
        if isinstance(self.scorer, dict):
            return list(self.scorer.keys())
        elif isinstance(self.scorer, Callable):
            return ['score']
        else:
            return []
        
    def estimate(self, X):
        """
        Returns the predicted probabilities for samples with confidence >= threshold.

        Parameters:
        -----------
        X : array-like
            Test features.

        Returns:
        --------
        array-like
            Predicted probabilities for samples meeting the confidence threshold.
        """
        proba = self.estimator.predict_proba(X)
        confidence = proba.max(axis=1)
        mask = confidence >= self.threshold
        return proba[mask]


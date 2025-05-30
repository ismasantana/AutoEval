from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
import numpy as np
from collections.abc import Callable # ALTERAR PARA CALLABLE()

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
        verbose : bool
            If True, prints additional information during evaluation.
        """
    
    def __init__(self, estimator, scorer=accuracy_score, threshold=0.8, limit_to_top_class=True, verbose=True):
        self.estimator = estimator
        self.scorer = scorer
        self.threshold = threshold
        self.limit_to_top_class = limit_to_top_class
        self.verbose = verbose

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def estimate(self, X):
        conf, correct = self._get_confidences_and_correct(X)
        if not np.any(correct):
            return {name: 0.0 for name in self._get_scorer_names()}

        y_pred = self.estimator.predict(X)
        y_estimated = [y_pred[i] if c == 1 else (y_pred[i]+1)%2 for i, c in enumerate(correct)]
        y_estimated = [int(y) for y in y_estimated]
        print(y_estimated, len(y_estimated), "\n")
        print(y_pred, len(y_pred))
        
        if isinstance(self.scorer, dict):
            return {
                name: func(y_estimated, y_pred)
                for name, func in self.scorer.items()
            }
        elif isinstance(self.scorer, Callable):
            return {'score': self.scorer(y_estimated, y_pred)}
        else:
            raise ValueError("'scorer' must be a callable or a dict of callables.")

    def _get_confidences_and_correct(self, X):
        """
        Computes confidence scores and applies the confidence threshold.

        Returns:
        --------
        tuple (confidences, correct)
            confidences: array of confidence scores
            correct: boolean array where confidence >= threshold
        """
        if hasattr(self.estimator, "predict_proba"):
            probas = self.estimator.predict_proba(X)
            conf = np.max(probas, axis=1) if self.limit_to_top_class else probas
        elif hasattr(self.estimator, "decision_function"):
            decision = self.estimator.decision_function(X)
            conf = np.max(decision, axis=1) if decision.ndim > 1 else np.abs(decision)
        else:
            raise ValueError("The estimator must implement predict_proba or decision_function.")
        
        correct = conf >= self.threshold
        return conf, correct

        
    def _get_scorer_names(self):
        """
        Returns the names of the scorers.
        
        Parameters:
        -----------
        None
        
        Returns:
        --------
        list
            List of scorer names.
        """
        if isinstance(self.scorer, dict):
            return list(self.scorer.keys())
        elif isinstance(self.scorer, Callable):
            return ['score']
        else:
            return []

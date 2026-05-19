import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class AdaCost(BaseEstimator, ClassifierMixin):
    """Cost-sensitive boosting classifier (Fan et al., 1999).

    Extends AdaBoost by weighting the update rule with a per-sample cost
    so that misclassifying the more dangerous class (webshell) is penalized
    more heavily.
    """

    def __init__(
        self,
        base_estimator=None,
        n_estimators: int = 100,
        learning_rate: float = 1.0,
        cost_positive: float = 1.0,
        cost_negative: float = 1.0,
        random_state: int = 42,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.cost_positive = cost_positive
        self.cost_negative = cost_negative
        self.random_state = random_state

    def _beta(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        beta = np.ones(len(y_true))
        fn_mask = (y_true == 1) & (y_pred == -1)
        fp_mask = (y_true == -1) & (y_pred == 1)
        beta[fn_mask] = self.cost_positive
        beta[fp_mask] = self.cost_negative
        return beta

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        if len(self.classes_) != 2:
            raise ValueError("AdaCost requires exactly 2 classes.")

        self.negative_class_ = self.classes_[0]
        self.positive_class_ = self.classes_[1]
        y_enc = np.where(y == self.positive_class_, 1, -1)

        n = X.shape[0]
        weights = np.full(n, 1.0 / n)

        base = self.base_estimator or DecisionTreeClassifier(max_depth=1)

        self.estimators_ = []
        self.alphas_ = []

        rng = np.random.RandomState(self.random_state)

        for t in range(self.n_estimators):
            est = clone(base)
            if hasattr(est, "random_state"):
                est.random_state = rng.randint(0, 2**31)

            est.fit(X, y_enc, sample_weight=weights)
            pred = est.predict(X)

            incorrect = (pred != y_enc).astype(float)
            error = np.dot(weights, incorrect)

            if error == 0.0:
                self.estimators_.append(est)
                self.alphas_.append(self.learning_rate * 10.0)
                break

            if error >= 0.5:
                break

            alpha = self.learning_rate * 0.5 * np.log((1.0 - error) / error)

            beta = self._beta(y_enc, pred)
            weights = weights * np.exp(alpha * incorrect * beta)
            weights /= weights.sum()

            self.estimators_.append(est)
            self.alphas_.append(alpha)

        if not self.estimators_:
            est = clone(base)
            if hasattr(est, "random_state"):
                est.random_state = self.random_state
            est.fit(X, y_enc, sample_weight=weights)
            self.estimators_.append(est)
            self.alphas_.append(1.0)

        return self

    def decision_function(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return sum(
            alpha * est.predict(X)
            for alpha, est in zip(self.alphas_, self.estimators_)
        )

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, self.positive_class_, self.negative_class_)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        proba_pos = 1.0 / (1.0 + np.exp(-2.0 * scores))
        return np.column_stack([1.0 - proba_pos, proba_pos])

    def get_params(self, deep=True):
        return {
            "base_estimator": self.base_estimator,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "cost_positive": self.cost_positive,
            "cost_negative": self.cost_negative,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)
        return self

import numpy as np
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr


DEFAULT_SCORERS = {
    "r2": r2_score,
    "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    "spearman": lambda y_true, y_pred: spearmanr(y_true, y_pred).statistic,  # type: ignore
}


def extend_slx(X_train, X_test, train_idx, test_idx, W):
    """
    engineer feature matrices with spatial lag features for slx modelling.
    Spatial lags for the test set are computed from training data only to avoid leakage.
    """

    WX_train = W[np.ix_(train_idx, train_idx)] @ X_train
    WX_test = W[np.ix_(test_idx, train_idx)] @ X_train
    return np.hstack([X_train, WX_train]), np.hstack([X_test, WX_test])


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model,
    cv_splitter,
    groups: np.ndarray | None = None,
    extend_func=None,
    compute_importance=None,
    scorers: dict | None = None,
) -> dict:
    if scorers is None:
        scorers = DEFAULT_SCORERS

    fold_scores = {name: [] for name in scorers}
    importance_per_fold = []

    split_args = (X, y, groups) if groups is not None else (X, y)

    for train_idx, test_idx in cv_splitter.split(*split_args):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if extend_func is not None:
            X_train, X_test = extend_func(X_train, X_test, train_idx, test_idx)

        m = clone(model)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)

        for name, scorer in scorers.items():
            fold_scores[name].append(scorer(y_test, y_pred))

        if compute_importance is not None:
            importance_per_fold.append(compute_importance(m))

    results = {}
    for name, scores in fold_scores.items():
        results[f"{name}_mean"] = float(np.mean(scores))
        results[f"{name}_std"] = float(np.std(scores))
        results[f"{name}_per_fold"] = scores

    if importance_per_fold:
        results["importance_per_fold"] = importance_per_fold

    return results

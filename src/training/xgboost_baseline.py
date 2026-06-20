"""
Stage 5 (part 2) — XGBoost baseline on geometry features.

This is the "stable Math baseline" that the spec wants to never perform
worse than. Two distinct fitting modes are exposed because they serve
different purposes and mixing them up causes label leakage:

    fit_oof(X, y)   -> Out-Of-Fold predictions on the TRAINING set, using
                       K-fold CV (each sample is predicted by a model that
                       never saw it during training). This is what you feed
                       to the residual DL head as the "frozen baseline
                       score" during training — using in-sample
                       (overfit) XGBoost predictions here would let the DL
                       branch learn from an overly-optimistic baseline and
                       hurt generalization at deployment time.

    fit_final(X, y) -> Fits ONE model on the full training set, for actual
                       deployment / test-set inference, where there is no
                       leakage concern (test set was never touched).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


DEFAULT_XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
)


class XGBoostBaseline:
    def __init__(self, xgb_params: Optional[dict] = None, n_splits: int = 5, random_state: int = 42):
        self.xgb_params = xgb_params or DEFAULT_XGB_PARAMS
        self.n_splits = n_splits
        self.random_state = random_state
        self.final_model: Optional[XGBClassifier] = None

    def fit_oof(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Returns (N,) out-of-fold P(Drowsy) for every training sample."""
        n = X.shape[0]
        oof_proba = np.zeros(n, dtype=np.float32)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            model = XGBClassifier(**self.xgb_params, random_state=self.random_state + fold_idx)
            model.fit(X[train_idx], y[train_idx])
            oof_proba[val_idx] = model.predict_proba(X[val_idx])[:, 1]

        return oof_proba

    def fit_final(self, X: np.ndarray, y: np.ndarray) -> "XGBoostBaseline":
        """Fits the deployment model on the FULL training set."""
        self.final_model = XGBClassifier(**self.xgb_params, random_state=self.random_state)
        self.final_model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(Drowsy) using the final model — call fit_final() first."""
        if self.final_model is None:
            raise RuntimeError("Gọi fit_final(X, y) trước khi predict_proba(). "
                                "fit_oof() chỉ dùng để sinh target cho residual head, không dùng để serve.")
        return self.final_model.predict_proba(X)[:, 1]

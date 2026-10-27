"""Item-based коллаборативная фильтрация.

sim(i, j) = <x_i, x_j> / (||x_i|| * ||x_j|| + shrink)   — косинус с shrinkage
Для каждого товара оставляем top-k соседей W[j, :]; score(u, j) = sum_{i in I(u)} x_ui * W[j, i].
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class ItemKNN:
    def __init__(self, k: int = 100, shrink: float = 10.0, weighting: str = "binary"):
        self.k, self.shrink, self.weighting = k, shrink, weighting

    def _weights(self, X: sp.csr_matrix) -> sp.csr_matrix:
        X = X.copy().astype(np.float32)
        if self.weighting == "binary":
            X.data[:] = 1.0
        elif self.weighting == "rating":
            pass  # оценка 1..5 как вес
        elif self.weighting == "idf":
            # редкие пользователи (с малой историей) дают более «информативный» сигнал
            n_u = np.diff(X.indptr)
            X.data[:] = np.repeat(np.log1p(X.shape[1] / np.maximum(n_u, 1)), n_u)
        else:
            raise ValueError(self.weighting)
        return X

    def fit(self, td):
        X = self._weights(td.X)
        G = (X.T @ X).toarray()
        norms = np.sqrt(np.diag(G))
        S = G / (np.outer(norms, norms) + self.shrink + 1e-8)
        np.fill_diagonal(S, 0.0)
        k = min(self.k, S.shape[1] - 1)
        top = np.argpartition(-S, k, axis=1)[:, :k]
        vals = np.take_along_axis(S, top, axis=1)
        rows = np.repeat(np.arange(S.shape[0]), k)
        W = sp.csr_matrix((vals.ravel(), (rows, top.ravel())), shape=S.shape, dtype=np.float32)
        self.WT = W.T.tocsr()  # score = X_u @ W^T
        self.X = X
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return (self.X[users] @ self.WT).toarray()

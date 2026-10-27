"""EASE^R (Steck, 2019).

min_B ||X - X B||_F^2 + lambda ||B||_F^2,  s.t. diag(B) = 0
Решение в замкнутом виде: P = (X^T X + lambda I)^{-1},  B_ij = -P_ij / P_jj,  B_jj = 0.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class EASE:
    def __init__(self, lam: float = 500.0, binary: bool = True):
        self.lam, self.binary = lam, binary

    def fit(self, td):
        X = td.X.copy().astype(np.float64)
        if self.binary:
            X.data[:] = 1.0
        G = (X.T @ X).toarray()
        G[np.diag_indices_from(G)] += self.lam
        P = np.linalg.inv(G)
        B = -P / np.diag(P)[None, :]
        np.fill_diagonal(B, 0.0)
        self.B = B.astype(np.float32)
        self.X = X.astype(np.float32).tocsr()
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.asarray(self.X[users] @ self.B)

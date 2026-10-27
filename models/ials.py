"""iALS — ALS для неявной обратной связи (Hu, Koren, Volinsky, 2008).

p_ui = 1, если взаимодействие наблюдалось, иначе 0;  c_ui = 1 + alpha * w_ui.
L = sum_{u,i} c_ui (p_ui - x_u^T y_i)^2 + reg (||X||^2 + ||Y||^2)
Шаг по пользователям (аналогично по товарам):
    (Y^T Y + Y^T (C_u - I) Y + reg I) x_u = Y^T C_u p_u
Систему решаем несколькими итерациями сопряжённых градиентов (Takács et al., 2011)
сразу для всех пользователей, используя разреженность C_u - I.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class IALS:
    def __init__(
        self,
        factors: int = 64,
        reg: float = 1.0,
        alpha: float = 10.0,
        epochs: int = 15,
        cg_steps: int = 3,
        confidence: str = "binary",
        seed: int = 42,
    ):
        self.factors, self.reg, self.alpha = factors, reg, alpha
        self.epochs, self.cg_steps, self.confidence, self.seed = epochs, cg_steps, confidence, seed

    def _conf(self, X: sp.csr_matrix) -> sp.csr_matrix:
        """Возвращает матрицу C - I (только наблюдаемые элементы)."""
        C = X.copy().astype(np.float32)
        if self.confidence == "binary":
            C.data[:] = self.alpha
        elif self.confidence == "rating":
            C.data = self.alpha * C.data / 5.0
        elif self.confidence == "log":
            C.data = self.alpha * np.log1p(C.data)
        else:
            raise ValueError(self.confidence)
        return C

    def _solve(self, Cm1: sp.csr_matrix, Y: np.ndarray, X0: np.ndarray) -> np.ndarray:
        YtY = Y.T @ Y
        rows = np.repeat(np.arange(Cm1.shape[0], dtype=np.int32), np.diff(Cm1.indptr))
        cols = Cm1.indices
        Yc = Y[cols]
        b = sp.csr_matrix((Cm1.data + 1.0, cols, Cm1.indptr), shape=Cm1.shape) @ Y

        def A(V):
            dots = np.einsum("nf,nf->n", Yc, V[rows])
            M = sp.csr_matrix((Cm1.data * dots, cols, Cm1.indptr), shape=Cm1.shape)
            return V @ YtY + self.reg * V + M @ Y

        x = X0.copy()
        r = b - A(x)
        p = r.copy()
        rs = np.einsum("nf,nf->n", r, r)
        for _ in range(self.cg_steps):
            Ap = A(p)
            a = rs / (np.einsum("nf,nf->n", p, Ap) + 1e-10)
            x += a[:, None] * p
            r -= a[:, None] * Ap
            rs_new = np.einsum("nf,nf->n", r, r)
            p = r + (rs_new / (rs + 1e-10))[:, None] * p
            rs = rs_new
        return x

    def fit(self, td):
        rng = np.random.default_rng(self.seed)
        Cu = self._conf(td.X)
        Ci = Cu.T.tocsr()
        self.U = (rng.standard_normal((td.n_users, self.factors)) * 0.01).astype(np.float32)
        self.V = (rng.standard_normal((td.n_items, self.factors)) * 0.01).astype(np.float32)
        for _ in range(self.epochs):
            self.U = self._solve(Cu, self.V, self.U).astype(np.float32)
            self.V = self._solve(Ci, self.U, self.V).astype(np.float32)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.U[users] @ self.V.T

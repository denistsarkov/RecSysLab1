"""Бейзлайн: популярность товара за последние `window_days` дней истории."""
from __future__ import annotations

import numpy as np


class Popularity:
    def __init__(self, window_days: float | None = 180):
        self.window_days = window_days

    def fit(self, td):
        df = td.df
        if self.window_days is not None:
            df = df[df["timestamp"] >= td.t_pred - self.window_days * 86400]
        cnt = np.bincount(df["iid"].values, minlength=td.n_items).astype(np.float32)
        self.pop = cnt + 1e-3 * np.asarray(td.X.getnnz(axis=0), dtype=np.float32)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.pop, (len(users), len(self.pop)))

"""Единый протокол оценки top-K для всех моделей."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import Encoder, TrainData


@dataclass
class EvalSet:
    users: np.ndarray  # uid пользователей, участвующих в оценке
    gt: sp.csr_matrix  # (len(users) x n_items), 1 = релевантный товар
    n_dropped_cold_users: int
    n_dropped_cold_items: int


def build_eval_set(enc: Encoder, td: TrainData, target: pd.DataFrame, pos_threshold: float = 4.0) -> EvalSet:
    """Релевантно = взаимодействие в целевом окне с оценкой >= pos_threshold.

    Холодные пользователи (нет истории) и холодные товары (нет взаимодействий в истории)
    исключаются: ни одна из пяти моделей не может их ранжировать честно.
    """
    t = enc.encode(target)
    t = t[t["rating"] >= pos_threshold]
    warm_users = np.asarray(td.X.getnnz(axis=1) > 0)
    warm_items = td.item_mask
    cold_u = ~warm_users[t["uid"].values]
    cold_i = ~warm_items[t["iid"].values]
    t = t[~cold_u & ~cold_i]
    users = np.sort(t["uid"].unique())
    row = pd.Series(np.arange(len(users)), index=users).loc[t["uid"]].values
    gt = sp.csr_matrix((np.ones(len(t), np.float32), (row, t["iid"].values)), shape=(len(users), td.n_items))
    gt.data[:] = 1.0
    return EvalSet(users, gt, int(cold_u.sum()), int((cold_i & ~cold_u).sum()))


def recommend(model, td: TrainData, users: np.ndarray, k: int = 10, batch_size: int = 2048) -> np.ndarray:
    """Top-k по всем товарам-кандидатам, кроме уже известных пользователю."""
    cand = td.item_mask
    out = np.empty((len(users), k), dtype=np.int64)
    for s in range(0, len(users), batch_size):
        ub = users[s : s + batch_size]
        scores = np.asarray(model.score(ub), dtype=np.float32).copy()
        scores[:, ~cand] = -np.inf
        seen = td.X[ub]
        scores[np.repeat(np.arange(len(ub)), np.diff(seen.indptr)), seen.indices] = -np.inf
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
        out[s : s + batch_size] = np.take_along_axis(top, order, axis=1)
    return out


def per_user_metrics(recs: np.ndarray, gt: sp.csr_matrix) -> pd.DataFrame:
    k = recs.shape[1]
    hits = np.zeros(recs.shape, dtype=np.float32)
    for r in range(recs.shape[0]):
        rel = gt.indices[gt.indptr[r] : gt.indptr[r + 1]]
        hits[r] = np.isin(recs[r], rel)
    n_rel = np.diff(gt.indptr)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * disc).sum(1)
    idcg = np.array([disc[: min(n, k)].sum() for n in n_rel])
    first = np.where(hits.any(1), hits.argmax(1) + 1, np.inf)
    return pd.DataFrame(
        {
            "ndcg": dcg / idcg,
            "recall": hits.sum(1) / np.minimum(n_rel, k),
            "hit_rate": hits.any(1).astype(np.float32),
            "mrr": 1.0 / first,
        }
    )


def evaluate(model, td: TrainData, es: EvalSet, k: int = 10, return_per_user: bool = False):
    recs = recommend(model, td, es.users, k)
    pu = per_user_metrics(recs, es.gt)
    res = {f"{m}@{k}": float(pu[m].mean()) for m in pu.columns}
    res[f"coverage@{k}"] = len(np.unique(recs)) / int(td.item_mask.sum())
    if return_per_user:
        return res, pu, recs
    return res


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """95% доверительный интервал для разности средних (a - b) по пользователям."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))

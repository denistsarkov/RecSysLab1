"""Метапризнаки для FM. Динамические признаки считаются point-in-time, без заглядывания в будущее."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import TrainData


def _platform(cats: list) -> str:
    return cats[1] if len(cats) > 1 else "na"


def _field_matrix(tokens: list[list[str]], vocab: dict[str, int] | None = None):
    """Мультихот-матрица поля; веса внутри поля нормированы на 1."""
    if vocab is None:
        vocab = {t: j for j, t in enumerate(sorted({t for ts in tokens for t in ts}))}
    rows, cols, vals = [], [], []
    for r, ts in enumerate(tokens):
        ts = [t for t in ts if t in vocab]
        for t in ts:
            rows.append(r)
            cols.append(vocab[t])
            vals.append(1.0 / len(ts))
    m = sp.csr_matrix((vals, (rows, cols)), shape=(len(tokens), len(vocab)), dtype=np.float32)
    return m, vocab


def _bucket(x: np.ndarray, edges, prefix: str) -> list[list[str]]:
    b = np.digitize(x, edges)
    return [[f"{prefix}:na"] if np.isnan(v) else [f"{prefix}:{k}"] for v, k in zip(x, b)]


def item_features(td: TrainData, min_store_items: int = 10) -> tuple[sp.csr_matrix, list[str]]:
    """Статичные поля товара: категории (все уровни пути), main_category, магазин/издатель,
    ценовой бакет, возрастной рейтинг, год выхода. Динамические (популярность, возраст)
    считаются на момент взаимодействия — см. ItemDynamics."""
    meta = td.item_meta
    fields = {}
    fields["category"] = [[f"cat:{c}" for c in cats[1:]] or ["cat:na"] for cats in meta["categories"]]
    fields["main_category"] = [[f"main:{c}"] for c in meta["main_category"].fillna("na")]
    store = meta["store"].fillna("na")
    vc = store.value_counts()
    store = store.where(store.map(vc) >= min_store_items, "other")
    fields["store"] = [[f"store:{s}"] for s in store]
    price = meta["price"].values.astype(float)
    fields["price"] = _bucket(price, np.nanquantile(price, [0.2, 0.4, 0.6, 0.8]), "price")
    rated = meta["rated"].fillna("na")
    rated = rated.where(rated.map(rated.value_counts()) >= 20, "other")
    fields["rated"] = [[f"rated:{r}"] for r in rated]
    fields["year"] = _bucket(meta["year"].values.astype(float), [2005, 2010, 2013, 2016, 2019, 2021], "year")

    mats, names = [], []
    for name, toks in fields.items():
        m, vocab = _field_matrix(toks)
        mats.append(m)
        names += list(vocab)
    return sp.hstack(mats).tocsr(), names


def user_features(td: TrainData) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Point-in-time признаки пользователя: профиль платформ (доли), бакет средней оценки и
    бакет активности — по истории строго ДО момента взаимодействия.

    Возвращает (F_pairs, F_now, names): F_pairs[r] — признаки для r-й строки td.df
    (на момент этого взаимодействия), F_now[u] — на момент t_pred (вся история).
    """
    df = td.df
    plat_names = td.item_meta["categories"].map(_platform).values
    plats = sorted(set(plat_names))
    pid = pd.Series(np.arange(len(plats)), index=plats).loc[plat_names].values
    mean_edges, act_edges = np.array([3.0, 4.0, 4.5, 4.99]), np.array([3, 4, 5, 6])
    n_mean, n_act = len(mean_edges) + 2, len(act_edges) + 2  # + «нет истории»
    names = [f"uplat:{p}" for p in plats] + [f"umean:{k}" for k in range(n_mean)] + [f"uact:{k}" for k in range(n_act)]

    def encode(counts, rsum, n):
        F = np.zeros((len(n), len(names)), np.float32)
        has = n > 0
        F[has, : len(plats)] = counts[has] / n[has, None]
        mb = np.where(has, np.digitize(rsum / np.maximum(n, 1), mean_edges), n_mean - 1)
        ab = np.where(has, np.digitize(np.log2(np.maximum(n, 1)), act_edges), n_act - 1)
        F[np.arange(len(n)), len(plats) + mb] = 1.0
        F[np.arange(len(n)), len(plats) + n_mean + ab] = 1.0
        return F

    order = np.lexsort((df["timestamp"].values, df["uid"].values))
    u, it, r = df["uid"].values[order], df["iid"].values[order], df["rating"].values[order]
    onehot = np.zeros((len(order), len(plats)), np.float32)
    onehot[np.arange(len(order)), pid[it]] = 1.0
    start = np.r_[True, u[1:] != u[:-1]]
    grp = np.cumsum(start) - 1
    # кумулятивные суммы внутри пользователя, исключая текущее взаимодействие
    def excl_cumsum(x):
        c = np.cumsum(x, axis=0)
        base = c[np.flatnonzero(start)][grp] - x[np.flatnonzero(start)][grp]
        return c - x - base
    counts = excl_cumsum(onehot)
    rsum = excl_cumsum(r.astype(np.float64))
    n = np.arange(len(order)) - np.flatnonzero(start)[grp]
    F_sorted = encode(counts, rsum, n.astype(np.float64))
    F_pairs = np.empty_like(F_sorted)
    F_pairs[order] = F_sorted
    # «сейчас»: вся история пользователя
    cnt_now = np.zeros((td.n_users, len(plats)), np.float32)
    np.add.at(cnt_now, (u, pid[it]), 1.0)
    rs_now = np.bincount(u, weights=r, minlength=td.n_users)
    n_now = np.bincount(u, minlength=td.n_users).astype(np.float64)
    return F_pairs, encode(cnt_now, rs_now, n_now), names


class ItemDynamics:
    """Point-in-time признаки товара в момент t (по истории строго до t):
    - pop: число взаимодействий за последние `window_days` дней (лог-бакеты);
    - age: сколько дней прошло с первого взаимодействия с товаром (бакеты, «нет» = ещё не было).
    Считаются одинаково для обучающих пар (t = время взаимодействия) и для рекомендаций (t = t_pred),
    поэтому модель не видит популярность, «накопленную» за счёт будущего.
    """

    POP_EDGES = np.arange(1, 8)
    AGE_EDGES = np.array([30, 90, 180, 365, 730, 1460])

    def __init__(self, td: TrainData, window_days: float = 90):
        df = td.df.sort_values(["iid", "timestamp"])
        self.big = np.int64(10**10)
        self.keys = df["iid"].values.astype(np.int64) * self.big + df["timestamp"].values.astype(np.int64)
        first = df.groupby("iid")["timestamp"].min().reindex(range(td.n_items))
        self.first = first.fillna(np.inf).values
        self.window = window_days * 86400
        self.n_pop = len(self.POP_EDGES) + 1
        self.n_age = len(self.AGE_EDGES) + 2  # + «ещё не появлялся»

    def __call__(self, iid: np.ndarray, t: np.ndarray):
        iid = iid.astype(np.int64)
        t = np.asarray(t, dtype=np.int64)
        hi = np.searchsorted(self.keys, iid * self.big + t, side="left")
        lo = np.searchsorted(self.keys, iid * self.big + t - int(self.window), side="left")
        pop = np.digitize(np.log1p(hi - lo), self.POP_EDGES)
        age_days = (t - self.first[iid]) / 86400.0
        age = np.where(age_days > 0, np.digitize(age_days, self.AGE_EDGES), self.n_age - 1)
        return pop, age

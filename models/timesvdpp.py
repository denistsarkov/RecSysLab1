"""timeSVD++ (Koren, 2009, «Collaborative Filtering with Temporal Dynamics»).

r̂_ui(t) = mu + b_u + alpha_u * dev_u(t) + b_{u,t} + b_i + b_{i,Bin(t)}
          + q_i^T ( p_u + alpha_pu * dev_u(t) + |N(u)|^{-1/2} sum_{j in N(u)} y_j )
dev_u(t) = sign(t - t_u) * |t - t_u|^beta,  t_u — средняя дата оценок пользователя (в днях).
Для t позже последней оценки пользователя дрейф не экстраполируем: t := min(t, t_last(u)).
b_{u,t} — смещение конкретного дня пользователя (для будущих дней неизвестно → 0). Почти у всех
пользователей одна оценка в день, поэтому b_{u,t} легко «запоминает» ответ; у него отдельная
(сильная) регуляризация reg_day.
Bin(t) — номер временного интервала; для момента рекомендации берём его бин
(после конца истории — последний бин).
L = sum (r_ui - r̂_ui)^2 + reg ||Θ_static||^2 + reg_time ||Θ_time||^2 + reg_day ||b_{u,t}||^2.
При use_time=False модель вырождается в SVD++ (для абляции временных компонент).
Для top-N ранжируем товары по r̂_ui(t_pred).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _csr_bag(indptr: np.ndarray, indices: np.ndarray, rows: np.ndarray):
    """Индексы и offsets для F.embedding_bag по выбранным строкам CSR."""
    starts, ends = indptr[rows], indptr[rows + 1]
    lens = ends - starts
    offsets = np.concatenate([[0], np.cumsum(lens)[:-1]])
    idx = np.repeat(starts - offsets, lens) + np.arange(lens.sum())
    return torch.from_numpy(indices[idx].astype(np.int64)), torch.from_numpy(offsets.astype(np.int64)), lens


class TimeSVDpp:
    def __init__(
        self,
        factors: int = 32,
        epochs: int = 20,
        lr: float = 3e-3,
        reg: float = 0.02,
        reg_time: float = 0.1,
        reg_day: float = 10.0,
        n_bins: int = 30,
        beta: float = 0.4,
        batch_size: int = 2048,
        use_time: bool = True,
        seed: int = 42,
    ):
        self.factors, self.epochs, self.lr, self.reg, self.reg_time = factors, epochs, lr, reg, reg_time
        self.reg_day = reg_day
        self.n_bins, self.beta, self.batch_size, self.use_time, self.seed = n_bins, beta, batch_size, use_time, seed

    def _dev(self, uid: np.ndarray, day: np.ndarray) -> np.ndarray:
        d = np.minimum(day, self.t_last[uid]) - self.t_u[uid]
        return (np.sign(d) * np.abs(d) ** self.beta / self.dev_scale).astype(np.float32)

    def _bin(self, day: np.ndarray) -> np.ndarray:
        b = ((day - self.day_min) / self.bin_width).astype(np.int64)
        return np.clip(b, 0, self.n_bins - 1)

    def fit(self, td):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        df = td.df
        uid, iid = df["uid"].values, df["iid"].values
        r = df["rating"].values.astype(np.float32)
        day = df["timestamp"].values / 86400.0
        nu, ni, f = td.n_users, td.n_items, self.factors

        self.mu = float(r.mean())
        self.t_u = np.zeros(nu)
        np.add.at(self.t_u, uid, day)
        self.t_u /= np.maximum(np.bincount(uid, minlength=nu), 1)
        self.t_last = np.full(nu, -np.inf)
        np.maximum.at(self.t_last, uid, day)
        self.day_min, day_max = day.min(), td.t_pred / 86400.0
        self.bin_width = (day_max - self.day_min) / self.n_bins + 1e-9
        self.dev_scale = 1.0
        self.dev_scale = float(np.std(self._dev(uid, day))) + 1e-8  # нормируем dev к единичному масштабу
        dev = self._dev(uid, day)
        bins = self._bin(day)
        ud = df["uid"].astype(np.int64) * 100000 + np.floor(day).astype(np.int64)
        ud_idx, _ = ud.factorize()

        # N(u): товары из истории пользователя (неявный сигнал), веса |N(u)|^-1/2
        X = td.X.tocsr()
        self.N_indptr, self.N_indices = X.indptr, X.indices
        n_u = np.diff(X.indptr)
        self.N_norm = (1.0 / np.sqrt(np.maximum(n_u, 1))).astype(np.float32)

        g = torch.Generator().manual_seed(self.seed)

        def P(*shape, std=0.01):
            return torch.nn.Parameter(torch.randn(*shape, generator=g) * std)

        def Z(*shape):
            return torch.nn.Parameter(torch.zeros(*shape))

        p = {"bu": Z(nu), "bi": Z(ni), "P": P(nu, f), "Q": P(ni, f), "Y": P(ni, f)}
        if self.use_time:
            p.update({"au": Z(nu), "but": Z(ud_idx.max() + 1), "bibin": Z(ni, self.n_bins), "ap": Z(nu, f)})
        self.p = p
        opt = torch.optim.Adam(p.values(), lr=self.lr)

        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
        self.history = []
        for _ in range(self.epochs):
            perm = rng.permutation(len(r))
            total = 0.0
            for s in range(0, len(perm), self.batch_size):
                b = perm[s : s + self.batch_size]
                u, i = t(uid[b]), t(iid[b])
                idx, offs, _ = _csr_bag(self.N_indptr, self.N_indices, uid[b])
                imp = F.embedding_bag(idx, p["Y"], offs, mode="sum") * t(self.N_norm[uid[b]])[:, None]
                pu = p["P"][u] + imp
                pred = self.mu + p["bu"][u] + p["bi"][i]
                reg = p["bu"][u] ** 2 + p["bi"][i] ** 2 + (p["P"][u] ** 2).sum(1) + (p["Q"][i] ** 2).sum(1)
                reg = reg + (imp**2).sum(1)
                treg = 0.0
                if self.use_time:
                    dv = t(dev[b])
                    bb = t(bins[b])
                    but = p["but"][t(ud_idx[b])]
                    bib = p["bibin"][i, bb]
                    pred = pred + p["au"][u] * dv + but + bib
                    pu = pu + p["ap"][u] * dv[:, None]
                    treg = p["au"][u] ** 2 + bib**2 + (p["ap"][u] ** 2).sum(1)
                    dreg = but**2
                pred = pred + (p["Q"][i] * pu).sum(1)
                mse = ((t(r[b]) - pred) ** 2).mean()
                loss = mse + self.reg * reg.mean()
                if self.use_time:
                    loss = loss + self.reg_time * treg.mean() + self.reg_day * dreg.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += mse.item() * len(b)
            self.history.append(total / len(r))
        self.t_pred_day = td.t_pred / 86400.0
        return self

    @torch.no_grad()
    def predict(self, uid: np.ndarray, iid: np.ndarray, day: np.ndarray) -> np.ndarray:
        """Предсказание оценок для пар (без b_{u,t}, если день не встречался)."""
        p = self.p
        uid, iid = np.asarray(uid, dtype=np.int64), np.asarray(iid, dtype=np.int64)
        idx, offs, _ = _csr_bag(self.N_indptr, self.N_indices, uid)
        u, i = torch.from_numpy(np.array(uid)), torch.from_numpy(np.array(iid))
        pu = p["P"][u] + F.embedding_bag(idx, p["Y"], offs, mode="sum") * torch.from_numpy(self.N_norm[uid])[:, None]
        pred = self.mu + p["bu"][u] + p["bi"][i]
        if self.use_time:
            dv = torch.from_numpy(self._dev(uid, day))
            pred = pred + p["au"][u] * dv + p["bibin"][i, torch.from_numpy(self._bin(day))]
            pu = pu + p["ap"][u] * dv[:, None]
        return (pred + (p["Q"][i] * pu).sum(1)).numpy()

    @torch.no_grad()
    def score(self, users: np.ndarray) -> np.ndarray:
        p = self.p
        idx, offs, _ = _csr_bag(self.N_indptr, self.N_indices, users)
        u = torch.from_numpy(users)
        pu = p["P"][u] + F.embedding_bag(idx, p["Y"], offs, mode="sum") * torch.from_numpy(self.N_norm[users])[:, None]
        user_bias = p["bu"][u]
        item_bias = p["bi"].clone()
        if self.use_time:
            day = np.full(len(users), self.t_pred_day)
            dv = torch.from_numpy(self._dev(users, day))
            user_bias = user_bias + p["au"][u] * dv
            pu = pu + p["ap"][u] * dv[:, None]
            item_bias = item_bias + p["bibin"][:, int(self._bin(np.array([self.t_pred_day]))[0])]
        return (self.mu + user_bias[:, None] + item_bias[None, :] + pu @ p["Q"].T).numpy()

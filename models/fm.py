"""Факторизационная машина (Rendle, 2010) с метапризнаками, обучение BPR.

y(x) = w0 + sum_k w_k x_k + sum_{k<l} <v_k, v_l> x_k x_l
     = w0 + sum_k w_k x_k + 1/2 * sum_f [ (sum_k v_kf x_k)^2 - sum_k v_kf^2 x_k^2 ]

Вектор x = [one-hot user id | признаки пользователя | one-hot item id | признаки товара].
Разделим его на пользовательскую (U) и товарную (I) части, s_U = sum_{k in U} v_k x_k.
Тогда y(u, i) = const(u) + lin(i) + self(i) + <s_U(u), s_I(i)>, где
lin(i) = sum_{k in I} w_k x_k, self(i) = 1/2 (||s_I||^2 - sum_{k in I} ||v_k||^2 x_k^2).
const(u) не влияет на ранжирование товаров для пользователя и сокращается в BPR:
L = -sum log sigma(y(u, i) - y(u, j)) + reg * (||s_U||^2 + ||s_I||^2 + ||s_J||^2),
где i — наблюдаемый товар, j — случайный негатив.

Динамические признаки товара (популярность за 90 дней и возраст) — признаки взаимодействия:
для обучающей пары они считаются на момент t этой пары (для негатива — на тот же t),
при рекомендации — на момент t_pred. Так нет утечки «будущей» популярности в обучение.
"""
from __future__ import annotations

import numpy as np
import torch

from utils.features import ItemDynamics, item_features, user_features


class FactorizationMachine:
    def __init__(
        self,
        factors: int = 64,
        lr: float = 2e-3,
        reg: float = 1e-4,
        epochs: int = 10,
        batch_size: int = 4096,
        use_meta: bool = True,
        use_ids: bool = True,
        use_dynamic: bool = True,
        use_user_meta: bool = True,
        seed: int = 42,
    ):
        self.factors, self.lr, self.reg, self.epochs = factors, lr, reg, epochs
        self.batch_size, self.use_meta, self.use_ids, self.seed = batch_size, use_meta, use_ids, seed
        self.use_dynamic = use_dynamic and use_meta
        self.use_user_meta = use_user_meta and use_meta

    def _build(self, td):
        nu, ni, f = td.n_users, td.n_items, self.factors
        if self.use_meta:
            Fi, self.item_feat_names = item_features(td)
            Fu_pairs, Fu, self.user_feat_names = user_features(td)
            if not self.use_user_meta:
                Fu_pairs, Fu = np.zeros((len(td.df), 0), np.float32), np.zeros((nu, 0), np.float32)
        else:
            Fu_pairs = np.zeros((len(td.df), 0), np.float32)
            Fi = np.zeros((ni, 0), np.float32)
            Fu = np.zeros((nu, 0), np.float32)
        Fi = Fi.toarray() if hasattr(Fi, "toarray") else Fi
        Fu = Fu.toarray() if hasattr(Fu, "toarray") else Fu
        self.Fi = torch.tensor(Fi, dtype=torch.float32)
        self.Fu = torch.tensor(Fu, dtype=torch.float32)  # на момент t_pred
        self.Fu_pairs = torch.tensor(Fu_pairs, dtype=torch.float32)  # на момент каждого взаимодействия
        g = torch.Generator().manual_seed(self.seed)

        def emb(n):
            return torch.nn.Parameter(torch.randn(n, f, generator=g) * 0.01)

        id_scale = 1.0 if self.use_ids else 0.0
        self.id_scale = id_scale
        self.params = {
            "V_uid": emb(nu),
            "V_iid": emb(ni),
            "V_uf": emb(Fu.shape[1]),
            "V_if": emb(Fi.shape[1]),
            "w_iid": torch.nn.Parameter(torch.zeros(ni)),
            "w_if": torch.nn.Parameter(torch.zeros(Fi.shape[1])),
        }
        if self.use_dynamic:
            self.dyn = ItemDynamics(td)
            for name, n in (("pop", self.dyn.n_pop), ("age", self.dyn.n_age)):
                self.params[f"V_{name}"] = emb(n)
                self.params[f"w_{name}"] = torch.nn.Parameter(torch.zeros(n))

    def _user_vec(self, u, Fu):
        p = self.params
        return self.id_scale * p["V_uid"][u] + Fu @ p["V_uf"]

    def _item_terms(self, i, t=None):
        p = self.params
        Fi = self.Fi[i]
        s = self.id_scale * p["V_iid"][i] + Fi @ p["V_if"]
        lin = self.id_scale * p["w_iid"][i] + Fi @ p["w_if"]
        sq = self.id_scale * (p["V_iid"][i] ** 2).sum(1) + (Fi**2) @ (p["V_if"] ** 2).sum(1)
        if self.use_dynamic:
            pop, age = (torch.from_numpy(a) for a in self.dyn(i.numpy(), t))
            for name, idx in (("pop", pop), ("age", age)):
                v = p[f"V_{name}"][idx]
                s = s + v
                lin = lin + p[f"w_{name}"][idx]
                sq = sq + (v**2).sum(1)
        return s, lin + 0.5 * ((s**2).sum(1) - sq)

    def fit(self, td):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self._build(td)
        opt = torch.optim.Adam(self.params.values(), lr=self.lr)
        users = td.df["uid"].values
        items = td.df["iid"].values
        times = td.df["timestamp"].values
        cand = np.flatnonzero(td.item_mask)
        self.history = []
        for _ in range(self.epochs):
            perm = rng.permutation(len(users))
            total = 0.0
            for s in range(0, len(perm), self.batch_size):
                b = perm[s : s + self.batch_size]
                u = torch.from_numpy(users[b])
                i = torch.from_numpy(items[b])
                j = torch.from_numpy(rng.choice(cand, size=len(b)))
                su = self._user_vec(u, self.Fu_pairs[torch.from_numpy(b)])
                tb = times[b]
                si, ti = self._item_terms(i, tb)
                sj, tj = self._item_terms(j, tb)
                diff = ti - tj + (su * (si - sj)).sum(1)
                loss = -torch.nn.functional.logsigmoid(diff).mean()
                loss = loss + self.reg * ((su**2).sum(1) + (si**2).sum(1) + (sj**2).sum(1)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            self.history.append(total / len(perm))
        with torch.no_grad():
            s, t = self._item_terms(torch.arange(td.n_items), np.full(td.n_items, td.t_pred))
            self.S_items, self.t_items = s.numpy(), t.numpy()
            self.S_users = self._user_vec(torch.arange(td.n_users), self.Fu).numpy()
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.S_users[users] @ self.S_items.T + self.t_items[None, :]

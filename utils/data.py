"""Загрузка, очистка, подвыборка и временное разбиение Amazon Reviews 2023 (Video_Games)."""
from __future__ import annotations

import json
import os
import random
import re
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

HF_BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"
FILES = {
    "Video_Games.csv": f"{HF_BASE}/benchmark/5core/rating_only/Video_Games.csv",
    "meta_Video_Games.jsonl": f"{HF_BASE}/raw/meta_categories/meta_Video_Games.jsonl",
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def download(raw_dir: str = "data/raw") -> None:
    """Скачивает рейтинги (5-core) и метаданные с HuggingFace, если их ещё нет."""
    os.makedirs(raw_dir, exist_ok=True)
    for name, url in FILES.items():
        path = os.path.join(raw_dir, name)
        if not os.path.exists(path):
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, path)


def load_ratings(raw_dir: str = "data/raw") -> pd.DataFrame:
    df = pd.read_csv(os.path.join(raw_dir, "Video_Games.csv"))
    df = df.rename(columns={"parent_asin": "item_id"})
    df["timestamp"] = (df["timestamp"] // 1000).astype(np.int64)  # мс -> с
    return df


def _parse_year(details: dict) -> float:
    for key in ("Release date", "Date First Available"):
        value = details.get(key)
        if value:
            m = re.search(r"(19|20)\d{2}", str(value))
            if m:
                return float(m.group(0))
    return np.nan


def load_meta(raw_dir: str = "data/raw", items: set | None = None) -> pd.DataFrame:
    """Читает метаданные товаров; оставляет только поля, известные заранее.

    average_rating / rating_number / Best Sellers Rank — снимок на 2023 год, т.е.
    утечка из будущего; их не используем как признаки.
    """
    rows = []
    with open(os.path.join(raw_dir, "meta_Video_Games.jsonl")) as f:
        for line in f:
            o = json.loads(line)
            if items is not None and o["parent_asin"] not in items:
                continue
            d = o.get("details") or {}
            price = o.get("price")
            try:
                price = float(price) if price is not None else np.nan
            except (TypeError, ValueError):
                price = np.nan
            rows.append(
                {
                    "item_id": o["parent_asin"],
                    "title": o.get("title"),
                    "main_category": o.get("main_category"),
                    "categories": list(o.get("categories") or []),
                    "store": o.get("store"),
                    "price": price,
                    "rated": d.get("Rated"),
                    "year": _parse_year(d),
                    "has_description": bool(o.get("description")),
                    "n_features": len(o.get("features") or []),
                }
            )
    return pd.DataFrame(rows)


def kcore(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """Итеративный k-core: повторяем фильтрацию, пока размеры не стабилизируются."""
    while True:
        n = len(df)
        df = df[df.groupby("item_id")["user_id"].transform("size") >= min_item]
        df = df[df.groupby("user_id")["item_id"].transform("size") >= min_user]
        if len(df) == n:
            return df.reset_index(drop=True)


def make_subsample(df: pd.DataFrame, start: str = "2012-01-01", min_user: int = 5, min_item: int = 10) -> pd.DataFrame:
    start_ts = int(pd.Timestamp(start).timestamp())
    df = df[df["timestamp"] >= start_ts]
    df = df.drop_duplicates(["user_id", "item_id"], keep="last")
    return kcore(df, min_user, min_item)


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    t_val: int
    t_test: int


def temporal_split(df: pd.DataFrame, val_q: float = 0.8, test_q: float = 0.9) -> Split:
    """Глобальное временное разбиение по квантилям времени взаимодействий."""
    t_val, t_test = (int(x) for x in df["timestamp"].quantile([val_q, test_q]).values)
    return Split(
        train=df[df["timestamp"] < t_val].reset_index(drop=True),
        val=df[(df["timestamp"] >= t_val) & (df["timestamp"] < t_test)].reset_index(drop=True),
        test=df[df["timestamp"] >= t_test].reset_index(drop=True),
        t_val=t_val,
        t_test=t_test,
    )


@dataclass
class TrainData:
    """Всё, что модель может видеть при обучении: только история до t_pred."""

    df: pd.DataFrame  # колонки uid, iid, rating, timestamp
    X: sp.csr_matrix  # users x items, значения = рейтинг
    n_users: int
    n_items: int
    item_meta: pd.DataFrame  # индекс = iid
    t_pred: int  # момент, в который строим рекомендации (начало целевого окна)

    @property
    def item_mask(self) -> np.ndarray:
        """Кандидаты: товары, у которых есть хотя бы одно взаимодействие в истории."""
        return np.asarray(self.X.getnnz(axis=0) > 0)


class Encoder:
    """Единая нумерация пользователей и товаров для всей подвыборки."""

    def __init__(self, df: pd.DataFrame, meta: pd.DataFrame):
        self.users = np.sort(df["user_id"].unique())
        self.items = np.sort(df["item_id"].unique())
        self.u2i = pd.Series(np.arange(len(self.users)), index=self.users)
        self.i2i = pd.Series(np.arange(len(self.items)), index=self.items)
        self.meta = meta.set_index("item_id").loc[self.items].reset_index()
        self.meta.index.name = "iid"

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(
            {
                "uid": self.u2i.loc[df["user_id"]].values,
                "iid": self.i2i.loc[df["item_id"]].values,
                "rating": df["rating"].values.astype(np.float32),
                "timestamp": df["timestamp"].values,
            }
        )
        return out

    def train_data(self, history: pd.DataFrame, t_pred: int) -> TrainData:
        h = self.encode(history)
        X = sp.csr_matrix(
            (h["rating"].values, (h["uid"].values, h["iid"].values)),
            shape=(len(self.users), len(self.items)),
            dtype=np.float32,
        )
        return TrainData(h, X, len(self.users), len(self.items), self.meta, t_pred)

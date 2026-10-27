"""Подбор гиперпараметров на validation через Optuna (TPE с фиксированным seed)."""
from __future__ import annotations

import time

import optuna
import pandas as pd

from .evaluation import evaluate

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune(model_cls, space, td, es, n_trials: int = 20, metric: str = "ndcg@10", seed: int = 42, fixed=None):
    """space(trial) -> dict гиперпараметров. Возвращает (best_params, таблица триалов)."""
    fixed = fixed or {}
    rows = []

    def objective(trial):
        params = {**space(trial), **fixed}
        t0 = time.time()
        model = model_cls(**params).fit(td)
        res = evaluate(model, td, es)
        rows.append({**params, **res, "time_s": time.time() - t0})
        return res[metric]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    trials = pd.DataFrame(rows)
    return {**study.best_params, **fixed}, trials

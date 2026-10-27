# ДЗ 1: рекомендательные системы — сравнение пяти моделей

**Студент:** Царьков Денис Владимирович

**Датасет:** Amazon Reviews 2023, категория **Video Games** (McAuley Lab, UCSD).
- Сайт: <https://amazon-reviews-2023.github.io/>
- Файлы (HuggingFace `McAuley-Lab/Amazon-Reviews-2023`):
  [`benchmark/5core/rating_only/Video_Games.csv`](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/blob/main/benchmark/5core/rating_only/Video_Games.csv),
  [`raw/meta_categories/meta_Video_Games.jsonl`](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/blob/main/raw/meta_categories/meta_Video_Games.jsonl)
- Статья: Hou et al., *Bridging Language and Items for Retrieval and Recommendation*, 2024.

Подвыборка: взаимодействия с 2012-01-01, затем итеративный k-core (пользователь ≥ 5, товар ≥ 10).
Получается 549 975 оценок, 67 007 пользователей и 11 192 товара.
Разбиение временное 80/10/10. Релевантно: оценка ≥ 4 в целевом окне.

## Результаты (test, top-10)

| модель | NDCG@10 | Recall@10 | HitRate@10 | Coverage@10 |
|---|---|---|---|---|
| ItemKNN | 0.0133 | 0.0216 | 0.0400 | 0.543 |
| iALS | 0.0126 | 0.0198 | 0.0388 | 0.010 |
| EASE | 0.0136 | 0.0227 | 0.0431 | 0.406 |
| **FM + метапризнаки** | **0.0250** | **0.0445** | **0.0761** | 0.038 |
| timeSVD++ | 0.0051 | 0.0094 | 0.0184 | 0.173 |
| Popularity (30 дней, бейзлайн) | 0.0172 | 0.0371 | 0.0653 | 0.002 |

Подробности, абляции и выводы — в [`experiments.ipynb`](experiments.ipynb).

## Структура

```
experiments.ipynb      основной ноутбук (выполнен)
models/
  item_knn.py          item-based CF, косинус с shrinkage
  ials.py              iALS (неявная обратная связь, CG-решатель)
  ease.py              EASE^R
  fm.py                факторизационная машина + BPR, point-in-time метапризнаки
  timesvdpp.py         timeSVD++
  popularity.py        бейзлайн
utils/
  data.py              загрузка, очистка, подвыборка, временное разбиение
  features.py          метапризнаки для FM
  evaluation.py        единый протокол top-K, метрики, бутстрэп
  tuning.py            подбор гиперпараметров (Optuna)
```

## Запуск

```bash
python -m venv .venv && source .venv/bin/activate  
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace experiments.ipynb
```



# Последняя модель CatBoost

Обучение: `../adaptive150_catboost.ipynb`. Политика кандидатов: `candidate_policy.json`. Индексы и кэш: `../retrieval_artifacts/`.

Адаптивный отбор до 150 кандидатов по типу запроса; 58 фичей, категориальный `query_type`. CatBoost YetiRank: depth=6, максимум 350 итераций, seed=42, early stopping на validation. Обучение использует 30 000 запросов; validation — 4 000, public test — выборка 10 000. Отсутствующие среди кандидатов позитивы исключены из обучающих групп, но учтены в итоговых метриках.

Validation: recall=0,7085, nDCG@20=0,419214. Public test: recall=0,6833, nDCG@20=0,413624, Hit@20=0,5575.

`catboost_ranker.cbm` — обученная модель; `feature_names.json` — порядок фичей; `feature_importance.csv` — их важности; `metrics.json` — параметры и результаты; `quantization_borders.tsv` — границы квантования. Остальные parquet/csv — результаты оценки и сырые top20 на validation/test-выборках. Готовый полный сабмит: `../submission_public.parquet`. Логи обучения удалены; при повторном обучении они появятся заново.

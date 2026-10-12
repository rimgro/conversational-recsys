# Music4All CRS: CatBoost baseline

Последняя модель: адаптивный отбор до 150 кандидатов и CatBoost с 58 фичами.

```text
music4all_crs/                   # исходные данные и описание датасета
build_candidate_indexes.ipynb    # пересборка BM25/TF-IDF-индексов
adaptive150_catboost.ipynb       # обучение и оценка CatBoost
submission_public.ipynb          # формирование public-сабмита
adaptive150_catboost_artifacts/  # модель, фичи, политика и метрики
retrieval_artifacts/             # индексы и кэш кандидатов для обучения
submission_public.parquet       # готовый файл для загрузки
submission_public_artifacts/    # проверки и подпись модели/запросов
requirements.txt                # зависимости Python
```

Описание исходных таблиц: [music4all_crs/README.md](music4all_crs/README.md).

Для нового окружения: `pip install -r requirements.txt`. Готовые индексы уже сохранены: можно сразу запускать обучение или формирование сабмита. `build_candidate_indexes.ipynb` нужен только при пересборке индексов. Ноутбуки определяют корень проекта при запуске из корня или папки датасета.

Обучение использует 30 000 запросов, validation — 4 000, public test для оценки — выборку 10 000. Кэш этих выборок сохранён в `retrieval_artifacts/candidate_cache/`. Validation nDCG@20: 0,4192; public-test nDCG@20: 0,4136.

Готовый `submission_public.parquet` содержит все 273 137 public-запросов, по 20 разных `m4a_id` и ответ про первый трек. Ответы составлены шаблонами по метаданным и истории пользователя, до 120 слов, без генеративной модели и public qrels. Для загрузки нужен только этот файл. Временные части удалены: повторное выполнение ячейки инференса рассчитает рекомендации заново.


## Аналитика генераторов

`candidate_generator_analysis.ipynb` — выполненный анализ вклада всех 11 источников на тех же 4 000 validation и 10 000 public-test запросов. Сравнивает самостоятельный recall, уникальные находки в union и абляции фиксированного top-150. Таблицы, примеры и графики: `candidate_analysis_artifacts/`. Модель и политика не изменены; test-метки используются только для диагностики.

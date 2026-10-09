# Ранкер (LightGBM LambdaRank)

Пул: hnsw + bm25_genres + bm25_tags + bm25_all (all = artist+song+genres+tags), по 100 → RRF (k=60) → top-200.
Группа = запрос, label 1 у таргета; группы без таргета в пуле не участвуют в обучении (в оценке дают 0).
Метрика nDCG@20 по `query_type` и `is_new`.

Данные — новый формат Music4All-CRS (`{split}_queries`, `{split}_qrels`, английские запросы, типы `novelty` и `similar_to`).
Обучение — на списке запросов из `select_queries.py` (100k, доли `query_type` как в train; `part` = train / valid,
valid — отдельные пользователи).

**Признаки «на момент запроса».** В train признаки пересчитываются на cutoff = начало дня `ts` запроса:
история до окна + позитивы этого пользователя в окне раньше cutoff; CTR — по событиям всех пользователей до cutoff.
В test все запросы видят один срез — начало тестового окна (ts запроса на тесте неизвестен).
Для `similar_to` референс X и треки его артиста выкидываются из групп (как в оценке).

## Модули

| модуль | что делает |
|---|---|
| `select_queries.py` | список запросов train для обучения (`query_id, user_id, query_type, source, part`) |
| `make_candidates.py` | кандидаты: BM25-индексы (+ hnsw файлом или функцией) → RRF → top-200 |
| `pit.py` | признаки на момент запроса (polars); `--tables-only` — только user и item таблицы |
| `fit.py` | обучение + скоринг test + отчёт ранкер vs RRF по `query_type` / `is_new` |
| `train.py`, `evaluate.py`, `ranker.py` | LightGBM (параметры, сохранение), метрики, `LGBMRanker` для пайплайна |
| `candidates.py` | источники (BM25 из индексов dev/bm25 или по HTTP, hnsw через функцию), RRF |
| `data.py`, `features.py`, `priors.py`, `run.py`, `recall.py` | v1 под старый (русский) формат датасета |

## Запуск

```bash
python -m ranker.select_queries --data-dir DATA --n 100000 --out ranker_train_queries.csv
python -m ranker.make_candidates --data-dir DATA --split train --queries ranker_train_queries.csv --bm25-index-dir INDEXES [--hnsw-candidates hnsw_train.parquet] --out cand_train.parquet
python -m ranker.make_candidates --data-dir DATA --split test_public --bm25-index-dir INDEXES [--hnsw-candidates hnsw_test.parquet] --out cand_test.parquet
python -m ranker.pit --data-dir DATA --split train --queries ranker_train_queries.csv --candidates cand_train.parquet --out f_train.parquet
python -m ranker.pit --data-dir DATA --split test_public --candidates cand_test.parquet --out f_test.parquet
python -m ranker.fit --train-features f_train.parquet --queries ranker_train_queries.csv --test-features f_test.parquet --data-dir DATA --model-dir model
```

Зависимости: polars, pyarrow, numpy, pandas, lightgbm. Модель сохраняется строкой (`model.txt`): LightGBM на Windows
не открывает пути с кириллицей.

## Признаки (16 + источники)

Считаются в `pit.build_chunk` (одна функция для train и test).
Колонка «Тип»: пользовательская, айтемная, кросс (пользователь × трек), источник (от кандидатогенерации), запрос × трек.
Колонка «Откуда»: решение из топ-4 RecSys Challenge 2026 (volart, niwatori, swyoo, team2_s2),
ветка `dev/base-structure` или **наш** (в топ-4 аналога нет).

| # | Тип | Признак | Что это | Откуда |
|---|---|---|---|---|
| 1 | источник | `rrf_score` | скор после RRF-слияния | base-structure; volart `rrf_rank` (признак №1 по важности), swyoo, team2 |
| 2 | источник | `n_sources` | сколько источников нашли трек | base-structure; niwatori, team2 |
| 3 | пользовательская | `age_bucket` | возраст по десятилетиям: 10s/20s/30s/40s/50+/unknown (<10 и >100 → unknown) | volart `age_*` |
| 4 | пользовательская | `gender` | m / f / unknown | volart `gender_*` |
| 5 | пользовательская | `top_genre` | самый слушаемый жанр (top-40 жанров + other / unknown) | **наш** |
| 6 | айтемная | `popularity` | `spotify_popularity` | volart, niwatori, team2 |
| 7 | айтемная | `ctr_item` | доля слушателей трека, слушавших его за 30 дней до cutoff, сглаживание α = 20 | **наш**; идея приоров — volart |
| 8 | айтемная | `ctr_artist` | то же для артиста | приоры + сглаживание — как volart (gold-frequency, MOVES-rate) |
| 9 | кросс | `age_at_release` | возраст пользователя в год выхода трека | niwatori `age_release_alignment` |
| 10 | кросс | `year_diff_history` | \|год трека − средний год истории\| | volart `year_diff_from_median_played` (у нас среднее) |
| 11 | кросс | `in_history` | слушал ли трек | base-structure; volart `is_in_played_history`, team2 `is_played` |
| 12 | кросс | `log_history_count` | log1p(сколько раз слушал трек) | base-structure |
| 13 | кросс | `artist_share` | доля прослушиваний истории на артиста | volart `played_artists_weighted`, niwatori `same_artist_history_count` |
| 14 | кросс | `log_album_listens` | log1p(прослушиваний альбома трека) | volart `same_album_as_any_played`, niwatori `same_album_history_count`, team2 `album_history_count` |
| 15 | кросс | `lang_share` | доля языка трека в истории | niwatori `user_lang_tag_overlap`, volart `culture_tag_overlap` |
| 16 | кросс | `genre_share` | доля прослушиваний истории в жанре трека (max по жанрам трека) | **наш** |
| 17 | кросс | `profile_tag_score` | вес тегов трека в профиле тегов пользователя | base-structure; volart `recent_tag_overlap` |
| 18 | запрос × трек | `query_tag_frac` | доля n-грамм запроса (теги/жанры каталога), которые есть у трека | base-structure; niwatori `tag_token_overlap` |

`query_type` — опционально (`fit.py --use-query-type`), по умолчанию не используется.

Сознательно не взяли из топ-4: rank/score каждого источника отдельно (niwatori), CTR альбома,
совпадение с последним прослушанным артистом/альбомом (volart, team2), co-occurrence (volart, №2 по важности),
иерархическую популярность (niwatori).

## Готовые таблицы (`ranker/artifacts/`)

| файл | что | ключ |
|---|---|---|
| `ranker_train_queries.csv` | 100k запросов train для обучения ранкера, `part` = train / valid | `query_id` |
| `user_features_{train,test_public}.parquet` | пользовательские признаки на момент запроса + `cutoff` | `query_id` |
| `item_features_{train,test_public}.parquet` | айтемные признаки всех треков на каждый `cutoff` (train: 30 дней окна, test: 20.02) | `cutoff`, `m4a_id` |

Склейка с кандидатами: user — по `query_id`, item — по (`cutoff` запроса из user-таблицы, `m4a_id`).
Кросс-признаки (пары запрос × кандидат) считаются `pit.py` после кандидатогенерации.

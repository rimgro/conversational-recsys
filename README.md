# Разговорная рекомендация музыки

Датасет **Music4All-CRS** (описание: [docs/dataset.md](docs/dataset.md)): каталог из 64k треков, история
прослушиваний Last.fm и текстовый профиль пользователя; на каждый трек, который пользователь послушал
в целевом месяце, есть синтетический запрос на английском. Задача: по запросу, истории и профилю найти этот трек
(nDCG@20) и ответить текстом; сабмит — `query_id`, `top20`, `response`.

Схема: `scheme.png`. Код — в пакете `recsys/`, ноутбуки только запускают его:

| Ноутбук | Для чего |
|---|---|
| `inference.ipynb` | запрос пользователя → ответ и треки; следующая реплика |
| `experiments.ipynb` | почему такая выдача: разбор запроса по шагам, как правила понимают настоящие запросы, сравнение вариантов конфига, LLM |
| `examples/candgen.ipynb` | проверка сервисов BM25 и HNSW руками: какие запросы и что они отвечают |

## Запуск

```bash
pip install -r requirements.txt
pytest -q                 # тесты, пайплайн целиком на синтетике
jupyter lab inference.ipynb   # или открыть в DataSphere
```

С сервисами на сервере: задать `BM25_URL`, `BM25_API_KEY`, `HNSW_URL`, `HNSW_API_KEY` (файл `.env` в корне — он не в git —
или секреты DataSphere) и поставить в ноутбуке `USE_CANDGEN = True`.

Данные — Music4All-CRS в папке `music4all_crs/` (`data.crs.dir`, в git не хранится): `tracks_meta.parquet`,
`train.parquet` / `test_public.parquet` (пользователи), `*_queries.parquet` (запросы), `*_qrels.parquet` (ответы).
Синтетика в формате датасета (`data.source: synthetic`) нужна только тестам: они не зависят от файлов.
В DataSphere: открыть ноутбук из корня репозитория; для LLM нужна GPU-конфигурация и `transformers`.

## Сервисы-кандгены

Сервисы разрабатываются и разворачиваются в своих ветках, здесь только клиент к ним (`recsys/retrieval/remote.py`)
и контракт [docs/candgen_api.md](docs/candgen_api.md):

| Ветка | Что | Сейчас |
|---|---|---|
| `dev/bm25` | сервис BM25 (`POST /bm25/search`, индексы genres, tags, title) | развёрнут на сервере |
| `fix/bm25-comma-split` | исправление сборки индексов BM25 (жанры через запятую) | ждёт слияния в `dev/bm25` |
| `dev/hnsw`, релиз `hnsw-v1` | текстовый семантический поиск (EmbeddingGemma + LanceDB, `POST /hnsw/search`) | развёрнут на сервере |
| `dev-ranker` | ранкер LightGBM | в разработке |

- `configs/candgen.yaml` включает источники `bm25_genres`, `bm25_tags` и `hnsw` с сервера вместо локального `bm25`.
  Адреса и ключи — `BM25_URL`, `BM25_API_KEY`, `HNSW_URL`, `HNSW_API_KEY` (файл `.env` или секреты DataSphere).
  Без неё всё считается локально, так что ноутбуки работают и без сервера.
- Если сервис недоступен, его источники возвращают пустой список с предупреждением, остальные работают.

## Валидация (`evaluate.py`)

Метрики пайплайна на сплите датасета: роль валидации играет `test_public` (12k пользователей, 273k запросов),
`test_private` скрыт. Метрика как в датасете — nDCG@20 с одной целью (для `similar_to` из выдачи убираются трек-образец
и его артист), плюс hit@20 и MRR — по типам запросов и по новым/знакомым трекам; recall каждого источника кандидатов
(`recall@<источник>`) и всех вместе (`recall@fused`) — качество кандгенов.

```bash
python evaluate.py --n-users 1000                   # быстро, ~3 мин локально
python evaluate.py --n-users all --candgen          # весь сплит, BM25 и HNSW с сервера
python evaluate.py --set ranker.type=heuristic      # любой параметр конфига
```

Результат — `outputs/<время>_<сплит>/`: `metrics.json` (метрики, конфиг, git-коммит, версии индексов сервисов,
ошибки сервисов), `submission.parquet` (сабмит), `by_query_type.csv`, `by_is_new.csv`, `per_request.csv`
(дописывается по ходу), `config.yaml`.
С `--candgen` прогон не начнётся, если сервис недоступен (код выхода 2).

Скорость: ~30 запросов/с локально, ~2 запроса/с с HNSW (он отвечает ~0.5 с); все пользователи по 5 запросов (~60k) —
около 35 минут локально, весь сплит (273k) — ~2.5 часа локально.
Короткие прогоны — из ноутбука (`!python evaluate.py ...`) или локально; полный — в DataSphere Jobs без открытого
ноутбука: `datasphere project job execute -p <id проекта> -c jobs/evaluate.yaml` (что подготовить — в начале файла).

## Что важно в данных

- **Одна цель на запрос**, 13 типов (`query_type`): 11 к прослушанным трекам (exact, lyrics_recall, lyrics_theme,
  genre, mood, situation, era_region, audio_attributes, complex, negative_constraint, vague_recall) и два discovery:
  `novelty` (новый артист, вне привычного вкуса) и `similar_to` («like X by Y but from other artists»).
  Метрики считаем по типам (`metrics_by`).
- **Большинство целей — повторные прослушивания** (`is_new = False`). Поэтому прослушанное по умолчанию не выкидываем
  (`fusion.exclude_listened: false`), а свою историю ищем отдельным источником `relisten`. Для `similar_to` и `novelty`
  разбор запроса ставит флаги «только новые треки» / «только новые артисты».
- **Запросы и теги на английском**, названия и имена — на языке оригинала.

## Пайплайн (`recsys/pipeline.py`)

| Шаг | Модуль | Сейчас |
|---|---|---|
| 1. разбор запроса → `DialogSummary` | `dialog.py` | правила (`rule`), `llm` с фолбэком на правила |
| 2. кандидаты | `retrieval/sources.py` | см. ниже |
| 3. RRF + фильтры + дедуп | `fusion.py` | настоящий |
| 4. ранкер | `ranking.py` | `stub` (порядок RRF), `heuristic`, `api` |
| 5. описание | `explain.py` | `stub` (приветствие + список), `llm` |

Источники кандидатов:

| Источник | Что делает | Для каких запросов |
|---|---|---|
| `bm25` | теги, жанры, артист, название, альбом, десятилетие, страна, язык | genre, mood, era_region, complex |
| `relisten` | треки из истории пользователя по совпадению с запросом и весу в истории | все, кроме discovery |
| `title` | артист + название по триграммам | exact |
| `lyrics` | строчка текста песни (выключен: нужен `data.crs.with_lyrics: true`) | lyrics_recall |
| `history` | новые треки по профилю тегов и артистов | novelty, `is_new` |
| `audio` | эмбеддинги MuQ, ближайшие к треку-образцу (similar_to) или к центру вкуса | similar_to, audio_attributes |
| `popular` | популярное в жанрах пользователя | холодный старт |
| `bm25_genres`, `bm25_tags`, `bm25_title` | BM25 с сервера (`type: bm25_api`); `bm25_title` ждёт индекса title | как `bm25` / `title` |
| `hnsw` | текстовый семантический поиск с сервера по реплике как есть (`type: hnsw_api`) | exact, lyrics_recall, vague_recall |

Контракты между шагами: `recsys/schemas.py` (`Request`, `DialogSummary`, `Candidate`, `FusedCandidate`, `RankedTrack`, `Response`).

## Структура

```
inference.ipynb            инференс: запрос -> ответ
evaluate.py                валидация: метрики на test_public -> outputs/
jobs/evaluate.yaml         то же в DataSphere Jobs
experiments.ipynb          эксперименты: разбор по шагам, сравнение вариантов
configs/default.yaml       все параметры и переключатели
configs/candgen.yaml       надстройка: BM25 и HNSW с сервера по HTTP
docs/dataset.md            описание датасета
docs/candgen_api.md        API сервисов-кандгенов (контракт с нашей частью)
examples/candgen.ipynb     проверка BM25 и HNSW руками
recsys/
  schemas.py      контракты между шагами
  config.py       загрузка YAML + overrides (предупреждает об опечатках в ключах)
  pipeline.py     пять шагов схемы
  text.py         нормализация тегов, токенизация, эпохи / страны / языки в запросе
  llm.py          StubLLM, LocalLLM (transformers), extract_json
  dialog.py       шаг 1: RuleSummarizer (заглушка), LLMSummarizer, промпты
  retrieval/
    bm25.py       BM25-индексы: теги, триграммы названий, тексты песен
    sources.py    шаг 2: локальные источники кандидатов
    remote.py     шаг 2: источники с сервера по HTTP (bm25_api, hnsw_api)
  fusion.py       шаг 3: RRF, фильтры, веса источников
  ranking.py      шаг 4: признаки, StubRanker, HeuristicRanker, APIRanker
  explain.py      шаг 5: StubExplainer, LLMExplainer (ответ по-английски)
  eval.py         nDCG@20 и др., evaluate, metrics_by, compare_configs
  data/
    catalog.py    каталог треков + эмбеддинги
    crs.py        чтение Music4All-CRS: tracks_meta -> Catalog, users + queries + qrels -> Request на query_id
    history.py    история пользователя -> профиль вкуса
    synthetic.py  синтетика в формате датасета
    load.py       load_data(cfg): synthetic | crs
tests/
```

## Как заменить заглушку

- **Ранкер**: `rank(features, ctx) -> DataFrame` с колонкой `rank_score`; признаки в `build_features` (`ranking.py`). Внешний сервис: `ranker.type: api`.
- **Новый источник кандидатов**: наследник `BaseRetriever` с `search(ctx) -> list[Candidate]` в `retrieval/`; зарегистрировать в `build_retrievers` (`retrieval/sources.py`) и включить в `configs/default.yaml`.
- **LLM**: `llm.type: local`, `summarizer.type: llm`, `explainer.type: llm`.

Метрики на синтетике завышены: запросы строятся из тех же тегов, по которым ищем. Они годятся для сравнения
вариантов и проверки, что код работает, но не для оценки реального качества.

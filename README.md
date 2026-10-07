# Разговорная рекомендация музыки

Датасет **Music4All-CRS** (описание: [docs/dataset.md](docs/dataset.md)): каталог из 64k треков, история
прослушиваний Last.fm и текстовый профиль пользователя; на каждый трек, который пользователь послушал
в целевом месяце, есть синтетический запрос на русском. Задача: по запросу, истории и профилю найти этот трек.

Схема: `scheme.png`. Главный файл: `main.ipynb`, остальное импортируется из пакета `recsys/`.

## Запуск

```bash
pip install -r requirements.txt
pytest -q                 # тесты, пайплайн целиком на синтетике (с FastAPI ещё и против настоящего candgen)
jupyter lab main.ipynb    # или открыть в DataSphere
```

С кандгенами: собрать индексы и поднять образ 1 (`candgen/RUN.md`), в ноутбуке поставить `USE_CANDGEN = True`.

Без данных всё работает на синтетике в формате датасета (`data.source: synthetic`). Для настоящих данных
положить `tracks_meta.parquet`, `train.parquet`, `test_public.parquet` в `data/` и поставить `data.source: crs`.
В DataSphere: открыть `main.ipynb` из корня репозитория; для LLM нужна GPU-конфигурация и `transformers`.

## Два образа

```
образ 1: candgen/  (bm25/ + hnsw/)          образ 2: recsys/ + main.ipynb  (Dockerfile в корне)
  POST /bm25/search  genres | tags | title    разбор запроса -> локальные источники (relisten, title,
  POST /hnsw/search  audio (MuQ)       <---     history, popular) + HTTP к образу 1 -> RRF -> ранкер -> ответ
  индексы строятся отдельно из tracks_meta
```

- **Образ 1** — кандгены, обучаются и обновляются отдельно, наружу только predict по HTTP. Сборка и API: [candgen/RUN.md](candgen/RUN.md).
- **Образ 2** — наша часть. Удалённые источники `bm25_genres`, `bm25_tags`, `bm25_title`, `hnsw_audio` (`recsys/retrieval/remote.py`)
  включаются надстройкой `configs/candgen.yaml`, адрес образа 1 — `CANDGEN_URL`. Без неё работают локальные `bm25` и `audio`,
  так что ноутбук запускается и без сервисов.
- Если образ 1 недоступен, удалённые источники возвращают пустой список с предупреждением, остальные работают.

## Что важно в данных

- **Одна цель на запрос**, 11 типов запросов (`query_family`): exact, lyrics_recall, genre, mood, situation, era_region, ...
  Метрики считаем по типам (`metrics_by`).
- **93% целей — повторные прослушивания** (`is_new = False`). Поэтому прослушанное не выкидываем
  (`fusion.exclude_listened: false`), а свою историю пользователя ищем отдельным источником `relisten`.
- **Запросы на русском, теги на английском.** Без LLM запрос разбирается словарём `recsys/ru.py`.

## Пайплайн (`recsys/pipeline.py`)

| Шаг | Модуль | Сейчас |
|---|---|---|
| 1. разбор запроса → `DialogSummary` | `dialog.py`, `ru.py` | правила (`rule`), `llm` с фолбэком на правила |
| 2. кандидаты | `retrieval/sources.py` | см. ниже |
| 3. RRF + фильтры + дедуп | `fusion.py` | настоящий |
| 4. ранкер | `ranking.py` | `stub` (порядок RRF), `heuristic`, `api` |
| 5. описание | `explain.py` | `stub` (приветствие + список), `llm` |

Источники кандидатов:

| Источник | Что делает | Для каких запросов |
|---|---|---|
| `bm25` | теги, жанры, артист, название, альбом, десятилетие, страна, язык | genre, mood, era_region, complex |
| `relisten` | треки из истории пользователя по совпадению с запросом и весу в истории | все: 93% целей отсюда |
| `title` | артист + название по триграммам, транслит кириллицы | exact |
| `lyrics` | строчка текста песни (выключен: нужен `data.crs.with_lyrics: true`) | lyrics_recall |
| `history` | новые треки по профилю тегов и артистов | discovery (`is_new`) |
| `audio` | эмбеддинги MuQ, ближайшие к центру вкуса | discovery, audio_attributes |
| `popular` | популярное в жанрах пользователя | холодный старт |
| `hnsw` | **заглушка** текстового семантического поиска (выключена) | — |
| `bm25_genres`, `bm25_tags`, `bm25_title` | BM25 из образа 1 (`type: bm25_api`) | как `bm25` / `title` |
| `hnsw_audio` | эмбеддинги из образа 1 (`type: hnsw_api`) | как `audio` |

Контракты между шагами: `recsys/schemas.py` (`Request`, `DialogSummary`, `Candidate`, `FusedCandidate`, `RankedTrack`, `Response`).

## Структура

```
main.ipynb                 главный ноутбук
Dockerfile                 образ 2 (наша часть)
configs/default.yaml       все параметры и переключатели
configs/candgen.yaml       надстройка: BM25 и HNSW из образа 1 по HTTP
bm25/                      кандген BM25: индекс, сборка, FastAPI (свой Dockerfile для отдельного запуска)
hnsw/                      кандген HNSW: индекс эмбеддингов, сборка, FastAPI-роутер
candgen/                   образ 1: bm25 + hnsw в одном сервисе (main.py, Dockerfile, compose, build_indexes.sh)
docs/dataset.md            описание датасета
examples/*.json            примеры запросов (id треков из синтетики)
recsys/
  schemas.py      контракты между шагами
  config.py       загрузка YAML + overrides (предупреждает об опечатках в ключах)
  pipeline.py     пять шагов схемы
  text.py         нормализация тегов, токенизация (латиница + кириллица)
  ru.py           русский словарь: основы слов -> теги, страны, языки, эпохи; транслит
  llm.py          StubLLM, LocalLLM (transformers), extract_json
  dialog.py       шаг 1: RuleSummarizer (заглушка), LLMSummarizer, промпты
  retrieval/
    bm25.py       BM25-индексы: теги, триграммы названий, тексты песен
    sources.py    шаг 2: локальные источники кандидатов
    remote.py     шаг 2: источники из образа 1 по HTTP (bm25_api, hnsw_api)
  fusion.py       шаг 3: RRF, фильтры, веса источников
  ranking.py      шаг 4: признаки, StubRanker, HeuristicRanker, APIRanker
  explain.py      шаг 5: StubExplainer, LLMExplainer (ответ по-русски)
  eval.py         метрики, evaluate, metrics_by, compare_configs
  data/
    catalog.py    каталог треков + эмбеддинги
    crs.py        чтение Music4All-CRS: tracks_meta -> Catalog, сплиты -> Request на каждый позитив
    history.py    история пользователя -> профиль вкуса
    synthetic.py  синтетика в формате датасета
    load.py       load_data(cfg): synthetic | crs
tests/
```

## Как заменить заглушку

- **Ранкер**: `rank(features, ctx) -> DataFrame` с колонкой `rank_score`; признаки в `build_features` (`ranking.py`). Внешний сервис: `ranker.type: api`.
- **Текстовый семантический поиск**: класс с `name = "hnsw"` и `search(ctx) -> list[Candidate]` (наследник `BaseRetriever`), лучше в отдельном файле `retrieval/hnsw.py`; зарегистрировать в `build_retrievers` (`retrieval/sources.py`).
- **LLM**: `llm.type: local`, `summarizer.type: llm`, `explainer.type: llm`.

Метрики на синтетике завышены: запросы строятся из тех же тегов, по которым ищем. Они годятся для сравнения
вариантов и проверки, что код работает, но не для оценки реального качества.

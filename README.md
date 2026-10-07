# Разговорная рекомендация музыки

Датасет **Music4All-CRS** (описание: [docs/dataset.md](docs/dataset.md)): каталог из 64k треков, история
прослушиваний Last.fm и текстовый профиль пользователя; на каждый трек, который пользователь послушал
в целевом месяце, есть синтетический запрос на русском. Задача: по запросу, истории и профилю найти этот трек.

Схема: `scheme.png`. Код — в пакете `recsys/`, ноутбуки только запускают его:

| Ноутбук | Для чего |
|---|---|
| `inference.ipynb` | запрос пользователя → ответ и треки; следующая реплика |
| `main.ipynb` | эксперименты: разбор шагов, метрики по типам запросов, сравнение вариантов, LLM |
| `examples/candgen.ipynb` | проверка сервисов BM25 и HNSW руками: какие запросы и что они отвечают |

## Запуск

```bash
pip install -r requirements.txt
pytest -q                 # тесты, пайплайн целиком на синтетике (с FastAPI ещё и против настоящих bm25/ и hnsw/)
jupyter lab inference.ipynb   # или открыть в DataSphere
```

С кандгенами на VPS: поднять их по [deploy/README.md](deploy/README.md), задать `BM25_URL` и `HNSW_URL`,
в ноутбуке поставить `USE_CANDGEN = True`.

Без данных всё работает на синтетике в формате датасета (`data.source: synthetic`). Настоящие данные:
`tracks_meta`, `train`, `test_public` — целиком (`train.parquet`) или частями (`train-00000-of-00078.parquet`, так
выложен полный датасет) в одной папке, например `data/full/`; в ноутбуке `DATA = "crs"`, `DATA_DIR = "data/full"`.
В DataSphere: открыть ноутбук из корня репозитория; для LLM нужна GPU-конфигурация и `transformers`.

## Кандгены на VPS

```
VPS                                                DataSphere: recsys/ + ноутбуки
  :8001  bm25/  POST /bm25/search  genres|tags|title   разбор запроса -> локальные источники (relisten, title,
  :8002  hnsw/  POST /hnsw/search  audio (MuQ)  <---     history, popular) + HTTP к VPS -> RRF -> ранкер -> ответ
  индексы строятся на VPS из tracks_meta
```

- **BM25 и HNSW** — два независимых сервиса (свой процесс, venv, порт, индексы): обучаются и обновляются отдельно,
  наружу только predict по HTTP. Запуск: [deploy/README.md](deploy/README.md).
- **Контракт** между ними и нашей частью: [docs/candgen_api.md](docs/candgen_api.md). Каждую из трёх частей можно менять
  независимо, пока он соблюдается.
- **Наша часть**: источники `bm25_genres`, `bm25_tags`, `bm25_title`, `hnsw_audio` (`recsys/retrieval/remote.py`) включаются
  надстройкой `configs/candgen.yaml`, адреса — `BM25_URL`, `HNSW_URL`. Тогда эмбеддинги в память DataSphere не грузятся.
  Без надстройки работают локальные `bm25` и `audio`, так что ноутбук запускается и без VPS.
- Если сервис недоступен, его источники возвращают пустой список с предупреждением, остальные работают.

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
| `bm25_genres`, `bm25_tags`, `bm25_title` | BM25 с VPS (`type: bm25_api`) | как `bm25` / `title` |
| `hnsw_audio` | эмбеддинги с VPS (`type: hnsw_api`) | как `audio` |

Контракты между шагами: `recsys/schemas.py` (`Request`, `DialogSummary`, `Candidate`, `FusedCandidate`, `RankedTrack`, `Response`).

## Структура

```
inference.ipynb            инференс: запрос -> ответ
main.ipynb                 эксперименты и метрики
configs/default.yaml       все параметры и переключатели
configs/candgen.yaml       надстройка: BM25 и HNSW с VPS по HTTP
bm25/                      кандген BM25: индекс, сборка, FastAPI
hnsw/                      кандген HNSW: индекс эмбеддингов, сборка, FastAPI
deploy/                    VPS: systemd-юниты, build_indexes.sh, run_local.sh (то же локально), check.py, инструкция
docs/dataset.md            описание датасета
docs/candgen_api.md        API кандгенов (контракт с нашей частью)
examples/candgen.ipynb     проверка BM25 и HNSW руками
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
    remote.py     шаг 2: источники с VPS по HTTP (bm25_api, hnsw_api)
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
- **Новый источник кандидатов**: наследник `BaseRetriever` с `search(ctx) -> list[Candidate]` в `retrieval/`; зарегистрировать в `build_retrievers` (`retrieval/sources.py`) и включить в `configs/default.yaml`.
- **LLM**: `llm.type: local`, `summarizer.type: llm`, `explainer.type: llm`.

Метрики на синтетике завышены: запросы строятся из тех же тегов, по которым ищем. Они годятся для сравнения
вариантов и проверки, что код работает, но не для оценки реального качества.

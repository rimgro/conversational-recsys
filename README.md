# Диалоговая рекомендация музыки

Вход: диалог (JSON, массив сообщений, на английском) + история прослушиваний + инфо о пользователе (текст).
Выход: top-k треков из каталога и текстовый ответ. Каталог: [Music4All-Onion](https://zenodo.org/records/6609677) + наш датасет с диалогами и названиями треков.

Схема: `scheme.png`. Главный файл: `main.ipynb`, остальное импортируется из пакета `recsys/`.

## Запуск

```bash
pip install -r requirements.txt
pytest -q                 # тесты, пайплайн целиком на синтетике
jupyter lab main.ipynb    # или открыть в DataSphere
```

В DataSphere: склонировать репозиторий в проект, открыть `main.ipynb` из корня репозитория (пути в конфиге относительные).
Для LLM нужна GPU-конфигурация и `transformers` (раскомментировать в `requirements.txt`).

## Пайплайн (`recsys/pipeline.py`)

| Шаг | Модуль | Реализации | Сейчас |
|---|---|---|---|
| 1. разбор диалога → `DialogSummary` | `dialog.py` | `rule`, `llm` (фолбэк на rule) | rule |
| 2. кандидаты | `retrieval/sources.py` | `bm25`, `hnsw` (**заглушка**), `history`, `popular` | все |
| 3. RRF + фильтры + дедуп | `fusion.py` | RRF с весами по контексту | настоящий |
| 4. ранкер | `ranking.py` | `stub` (порядок RRF), `heuristic`, `api` | stub |
| 5. описание | `explain.py` | `stub` (приветствие + список), `llm` | stub |

Контракты между шагами: `recsys/schemas.py` (`Request`, `DialogSummary`, `Candidate`, `FusedCandidate`, `RankedTrack`, `Response`).

- **История** участвует трижды: источник `history` (профиль тегов и артистов → BM25; потом заменить на collab-HNSW), `popular` в жанрах пользователя, признаки ранкера. Прослушанное, показанное и скипнутое отфильтровывается.
- **Исключения** из диалога (`no rap`, `without female vocalists`) работают как жёсткий фильтр на шаге 3.
- **Несколько реплик**: `request.next_turn(response, "more energetic")` добавляет ответ и показанные треки в запрос.

## Данные (`data.source` в `configs/default.yaml`)

| source | Что нужно | Запросы для оценки |
|---|---|---|
| `synthetic` | ничего | синтетические диалоги с целями |
| `onion` | файлы Onion в `data.onion.dir` (`id_tags_dict`, `id_genres_tf-idf`, `userid_trackid_count`) | шаблонные запросы из тегов отложенных треков |
| `dataset` | наш датасет в `data.dataset.dir` | диалоги датасета |

**Формат нашего датасета пока предполагаемый** (описан в `recsys/data/dataset.py`): `dialogs.jsonl` с полями
`dialog_id, user_id, user_info, history, messages[{role, text}], target_track_ids` и `tracks.*` с `track_id, title, artist`.
Если поля называются иначе, поменять `data.dataset.fields` в конфиге. Теги можно взять из Onion: `use_onion_tags: true`.

Пример формата: `examples/request_01.json`. Сохранить синтетику в формате датасета: `data.synthetic.save_dir`.

## Структура

```
main.ipynb                 главный ноутбук
configs/default.yaml       все параметры и переключатели заглушек
examples/*.json            примеры запросов
recsys/
  schemas.py      контракты между шагами (Request, DialogSummary, Candidate, ..., Response)
  config.py       загрузка YAML + overrides (предупреждает об опечатках в ключах)
  pipeline.py     пять шагов схемы
  text.py         нормализация тегов, токенизация
  llm.py          StubLLM, LocalLLM (transformers), extract_json
  dialog.py       шаг 1: RuleSummarizer (заглушка), LLMSummarizer, промпты
  retrieval/
    bm25.py       BM25-индекс по тегам/жанрам/артисту/названию
    sources.py    шаг 2: источники bm25, history, popular, hnsw (заглушка)
  fusion.py       шаг 3: RRF, фильтры, веса источников
  ranking.py      шаг 4: признаки, StubRanker, HeuristicRanker, APIRanker, не больше N треков артиста
  explain.py      шаг 5: StubExplainer, LLMExplainer
  eval.py         метрики, evaluate, compare_configs
  data/
    catalog.py    каталог треков
    loaders.py    чтение файлов Onion и таблиц метаданных
    history.py    история пользователя -> профиль вкуса
    dataset.py    адаптер нашего датасета с диалогами
    synthetic.py  синтетические данные и запросы
    load.py       load_data(cfg): synthetic | onion | dataset
tests/
```

## Как заменить заглушку

- **HNSW**: класс с `name = "hnsw"` и `search(ctx) -> list[Candidate]` (наследник `BaseRetriever`), лучше в отдельном файле `retrieval/hnsw.py`; зарегистрировать в `build_retrievers` (`retrieval/sources.py`).
- **Ранкер**: `rank(features, ctx) -> DataFrame` с колонкой `rank_score`; признаки в `build_features` (`ranking.py`). Внешний сервис: `ranker.type: api`.
- **LLM**: `llm.type: local`, `summarizer.type: llm`, `explainer.type: llm`.

Метрики на `synthetic` и `onion` завышены: запросы строятся из тегов целей, а BM25 ищет по тем же тегам. Они годятся для сравнения вариантов, но не для оценки реального качества.

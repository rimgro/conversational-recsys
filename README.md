# Диалоговая рекомендация музыки

Вход: диалог (JSON, массив сообщений, на английском) + история прослушиваний + инфо о пользователе (текст).
Выход: top-k треков из каталога и текстовый ответ. Каталог: [Music4All-Onion](https://zenodo.org/records/6609677) + наш датасет с диалогами и названиями треков.

Схема: `scheme.png`. Главный файл: `main.ipynb`, остальное импортируется из пакета `recsys/`.

## Запуск

```bash
pip install -r requirements.txt
pytest -q                 # 12 тестов, пайплайн целиком на синтетике
jupyter lab main.ipynb    # или открыть в DataSphere
```

В DataSphere: склонировать репозиторий в проект, открыть `main.ipynb` из корня репозитория (пути в конфиге относительные).
Для LLM нужна GPU-конфигурация и `transformers` (раскомментировать в `requirements.txt`).

## Пайплайн (`recsys/pipeline.py`)

| Шаг | Модуль | Реализации | Сейчас |
|---|---|---|---|
| 1. разбор диалога → `DialogSummary` | `dialog/` | `rule`, `llm` (фолбэк на rule) | rule |
| 2. кандидаты | `retrieval/` | `bm25`, `hnsw` (**заглушка**), `history`, `popular` | все |
| 3. RRF + фильтры + дедуп | `fusion/` | RRF с весами по контексту | настоящий |
| 4. ранкер | `ranking/` | `stub` (порядок RRF), `heuristic`, `api` | stub |
| 5. описание | `explain/` | `stub` (приветствие + список), `llm` | stub |

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
examples/request_01.json   пример запроса
recsys/
  schemas.py  config.py  pipeline.py  text.py
  data/       catalog, loaders (Onion), history (профиль), dataset (наш формат), synthetic, load
  llm/        StubLLM, LocalLLM (transformers), extract_json
  dialog/     rule.py, llm_summarizer.py, prompts.py
  retrieval/  bm25.py, hnsw_stub.py, history.py, popular.py
  fusion/     rrf.py, filters.py
  ranking/    features.py, stub.py, heuristic.py, api.py, diversity.py
  explain/    stub.py, llm_explainer.py
  eval/       metrics.py, evaluate.py (evaluate, compare_configs)
tests/
```

## Как заменить заглушку

- **HNSW**: класс с `name = "hnsw"` и `search(ctx) -> list[Candidate]` (наследник `BaseRetriever`), зарегистрировать в `retrieval/__init__.py`.
- **Ранкер**: `rank(features, ctx) -> DataFrame` с колонкой `rank_score`; признаки в `ranking/features.py`. Внешний сервис: `ranker.type: api`.
- **LLM**: `llm.type: local`, `summarizer.type: llm`, `explainer.type: llm`.

Метрики на `synthetic` и `onion` завышены: запросы строятся из тегов целей, а BM25 ищет по тем же тегам. Они годятся для сравнения вариантов, но не для оценки реального качества.

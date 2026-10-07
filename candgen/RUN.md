# Образ 1: кандидат-генераторы (BM25 + HNSW)

Один HTTP-сервис с двумя кандгенами. Код кандгенов лежит в `bm25/` и `hnsw/`, здесь только сборка
(`main.py` подключает маршруты обоих). Наша часть (образ 2, пакет `recsys`) вызывает их по API.

| Метод | Тело | Ответ |
|---|---|---|
| `POST /bm25/search` | `{"words": ["rock", "indie"], "k": 200, "exclude_ids": [], "index": "genres"}` | `{"ids", "scores", "index", "index_version"}` |
| `POST /hnsw/search` | `{"track_ids": [...], "weights": [...], "k": 200, "exclude_ids": [], "index": "audio"}` или `{"vector": [...], ...}` | `{"ids", "scores", "index", "index_version", "n_query_tracks"}` |
| `GET /health`, `GET /hnsw/health` | | загруженные индексы и их версии |

Индексы BM25: `genres` (по умолчанию), `tags`, `title` (артист + название). Индекс HNSW: `audio` (эмбеддинги MuQ).
Подробности BM25: `bm25/RUN.md`.

## Индексы

Строятся отдельно от сервиса, из `tracks_meta.parquet`:

```bash
sh candgen/build_indexes.sh data/tracks_meta.parquet indexes          # локально, из корня репозитория
cd candgen && docker compose run --rm build-indexes                   # то же в контейнере
```

Результат: `indexes/bm25/{genres,tags,title}`, `indexes/hnsw/audio`. Сервис читает их один раз при старте.

## Запуск

```bash
docker build -f candgen/Dockerfile -t recsys-candgen .                # из корня репозитория
cd candgen && docker compose up --build                               # или через compose
```

Без Docker (Python 3.10+): `BM25_INDEXES_DIR=indexes/bm25 HNSW_INDEXES_DIR=indexes/hnsw uvicorn main:app --app-dir candgen --port 8000`.

## Env

| Переменная | Значение |
|---|---|
| `BM25_INDEXES_DIR` | папка индексов BM25 (`/indexes/bm25` в образе), обязательна, должен быть индекс `genres` |
| `HNSW_INDEXES_DIR` | папка индексов HNSW (`/indexes/hnsw`); если пусто, `/hnsw/search` отвечает 503 |
| `BM25_API_KEY` | если задан, нужен заголовок `X-API-Key` (для HNSW тоже, если не задан `HNSW_API_KEY`) |

## Известная проблема

`bm25/build_index.py` не делит строки вида `"rock,hard rock"` по запятым: в индекс попадают слова
`rock,hard`, и запрос `hard` их не находит. В `tracks_meta.parquet` жанры и теги хранятся именно так.
Тест `candgen/tests/test_gateway.py::test_bm25_splits_comma_separated_tags` помечен xfail до исправления.

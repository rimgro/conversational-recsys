# HNSW / LanceDB: индексация Music4All-CRS через EmbeddingGemma-2

Пайплайн ретрива: `tracks_meta` → текст трека (метаданные + лирика) →
эмбеддинги **`google/embeddinggemma-2`** (768-d, mean pooling, L2-нормализация) →
**LanceDB** (cosine, `IVF_HNSW_SQ`) → оценка поиска по синтетическим запросам из `train`.

- Код: [`recsys/retrieval/gemma_lancedb.py`](../recsys/retrieval/gemma_lancedb.py) (standalone: CLI + импорт).
- Kaggle-ноутбук: [`notebooks/kaggle_gemma_lancedb.ipynb`](../notebooks/kaggle_gemma_lancedb.ipynb) (одно-ячеечный, рабочий).
- Готовый индекс: [`artifacts/lancedb.zip`](../artifacts/lancedb.zip) (Git LFS).

> Модуль самодостаточен и не меняет текущий `recsys/pipeline.py`. Подключение в
> пайплайн как источника кандидатов — отдельный шаг (см. §8).

---

## 0. Требования

- Python 3.12, `pip`/`uv`.
- GPU для реального эмбеддинга (Kaggle T4x2 достаточно), интернет.
- Датасет CRS (папка с `tracks_meta-*.parquet`, `train-*.parquet`, …).
- Пакеты: `torch`, `transformers>=5.19`, `sentence-transformers==6.1.0`,
  `lancedb`, `pyarrow`, `numpy`.

---

## 1. Данные

Датасет **Music4All-CRS** — 64 016 треков, 12 891 пользователь (~483k позитивов
с синтетическими запросами). Ключ трека — `m4a_id`.

Для Kaggle датасет загружается как приватный датасет и монтируется в
**`/kaggle/input/datasets/<user>/<slug>`** (обратите внимание на префикс `/datasets/`).

---

## 2. Локальный smoke-прогон (без GPU)

```bash
pip install lancedb pyarrow numpy

DS="path/to/CRS dataset"          # папка с tracks_meta-*.parquet
python -m recsys.retrieval.gemma_lancedb --work-dir data/crs --dataset "$DS" build-texts
python -m recsys.retrieval.gemma_lancedb --work-dir data/crs --dataset "$DS" embed --embedder hashing --hash-dim 256
python -m recsys.retrieval.gemma_lancedb --work-dir data/crs --dataset "$DS" index
python -m recsys.retrieval.gemma_lancedb --work-dir data/crs --dataset "$DS" evaluate --embedder hashing --max-queries 500
```

`--embedder hashing` — не семантический, только проверка механики.

---

## 3. Полная индексация с нуля (GPU / Kaggle)

Проверенные на T4 параметры: **`--dtype float16 --batch-size 8 --max-seq-length 1024`**
(иначе OOM / NaN).

```bash
pip install -q "sentence-transformers==6.1.0" "transformers==5.19.0" lancedb

DS="/kaggle/input/datasets/<user>/<slug>"
WORK="/kaggle/working/crs_data"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m recsys.retrieval.gemma_lancedb --work-dir $WORK --dataset $DS build-texts
python -m recsys.retrieval.gemma_lancedb --work-dir $WORK --dataset $DS embed \
  --embedder gemma --device cuda --dtype float16 \
  --batch-size 8 --max-seq-length 1024 --shard-size 10000
python -m recsys.retrieval.gemma_lancedb --work-dir $WORK --dataset $DS index
python -m recsys.retrieval.gemma_lancedb --work-dir $WORK --dataset $DS evaluate \
  --embedder gemma --device cuda --dtype float16 --max-queries 5000 --batch-size 16
```

Артефакты: `$WORK/tracks_texts.jsonl`, `$WORK/vectors/all.npz` (+ шарды),
`$WORK/lancedb/`, `$WORK/eval_query_gemma.json`.

Если модуль запускается как файл (не как пакет): `python recsys/retrieval/gemma_lancedb.py ...`.

---

## 4. Готовый LanceDB

`artifacts/lancedb.zip` (Git LFS) → распаковать и запрашивать:

```bash
git lfs install && git lfs pull
unzip artifacts/lancedb.zip -d lancedb      # → lancedb/tracks.lance
```

```python
import lancedb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-2", device="cuda")
q = model.encode("task: search result | query: <запрос>", normalize_embeddings=True)

tbl = lancedb.connect("lancedb").open_table("tracks")
rows = (tbl.search(q, vector_column_name="vector_combined")
           .metric("cosine").limit(20).to_list())
for r in rows:
    print(round(1 - r["_distance"], 3), r["title"], "—", r["artist"])
```

Таблица `tracks`: **64 016 строк**, колонки
`id, spotify_id, title, artist, album, year, lang, genres, tags, document, vector_combined (768d)`.

---

## 5. Результаты (5 000 запросов из `train`)

| Метрика | query-only | +профиль |
|---|---|---|
| Recall@20 | **0.167** | 0.040 |
| Recall@100 | **0.226** | 0.085 |
| nDCG@20 | **0.132** | 0.022 |
| MRR | **0.123** | 0.018 |

`query_family` (query-only, Recall@100): `exact` **0.95** (MRR 0.90),
`vague_recall` 0.33, `complex` 0.30, `lyrics_recall` 0.29, `era_region` 0.20,
`lyrics_theme` 0.19, `genre` 0.12; слабо — `mood`/`situation`/`audio_attributes`
(~0–0.04). Полный отчёт — `eval_query_gemma.json`.

Интерпретация: пайплайн исправен (`exact` почти идеален), но агрегат — сырой
zero-shot бейзлайн. Потенциал роста — в построении текста/запроса и
дообучении, не в модели.

---

## 6. Грабли (проверено)

- **OOM**: `batch=64/seq=2048` не влезает в T4 → `batch=8/seq=1024`.
- **NaN в векторах** (fp16 overflow в mean-pooling) ломает LanceDB → в коде
  стоит `nan_to_num` (и в энкодере, и перед записью в LanceDB).
- **Профиль в промпте вредит**: длинный `user_profile` размывает запрос
  (R@100 0.226 → 0.085). Использовать только запрос.
- **Kaggle**: одновременно — только **1 GPU-сессия**; многоячеечные ноутбуки
  иногда дают ERROR с пустым логом → использовать одно-ячеечный ноутбук.
- Путь датасета на Kaggle — `/kaggle/input/datasets/<user>/<slug>`.
- Кросс-языковость: запросы RU, документы EN — часть провала по mood/genre.

---

## 7. Как формируется текст трека

Документ (без `Instruct:`):

```
title: {title} | text:
Artist: ...; Album: ... (year); Genres: ...; Tags: ...; Language: ...;
Description: ...; Artist bio: ...; Album info: ...;
Audio: tempo ...; key ...; energy ...; valence ...; danceability ...; duration ...; instrumental ...;
Listeners: popularity ...; listeners ...;
Lyrics: ...
```

Запрос: `task: search result | query: {запрос}`.

---

## 8. Интеграция в пайплайн (следующий шаг)

Модуль даёт `search_lancedb(...)` и `GemmaEmbedder`; чтобы использовать как
источник кандидатов в `recsys/pipeline.py`, нужно обернуть поиск в интерфейс
источника (как `recsys/retrieval/sources.py`) и добавить список в fusion (RRF).
Пока модуль не подключён к общему пайплайну — это отдельная задача.

# Qwen history-aware candidate generator

Две совместно обучаемые башни для Music4All-CRS: **Qwen3-Embedding-0.6B + LoRA + история прослушиваний** кодирует запрос, **Transformer признаков трека** кодирует каталог. На выходе — нормированные векторы **256** и согласованный HNSW. Кандген возвращает до **200 canonical-кандидатов**.

Эта ветка содержит код кандгена, обучающегося на G4 RTX PRO 6000 Blackwell. Ядро `recsys/` совпадает побайтно с кодом текущего checkpoint: [SHA256 manifest](docs/training_code_manifest.json). Управление Colab, оценка топ-200 и политика остановки находятся в `jobs/`, чтобы сохранить совместимость checkpoint. Веса, датасет, кэш и OAuth-файлы хранятся отдельно.

## Структура

| Путь | Назначение |
|---|---|
| `recsys/two_tower_qwen/` | история, Qwen LoRA, sampled softmax, trainer, resume, экспорт и инференс |
| `recsys/track_tower.py` | Transformer пяти блоков признаков трека |
| `configs/qwen_g4.json` | параметры текущей архитектуры, micro32/effective32 |
| `tools/train_qwen.py` | обычный запуск на GPU, resume и экспорт |
| `tools/predict_qwen.py` | CLI exact/HNSW с canonical-дедупликацией и топ-200 |
| `jobs/colab_continue/` | рабочий Colab runner: ещё пять эпох, smoke/rollback, бэкапы и early stopping |
| `jobs/colab_recall/` | holdout, Recall/nDCG/MRR, выдачи exact/HNSW |
| `tests/` | CPU-проверки модели, градиентов, resume/RNG, загрузки и экспорта |

Архитектура: [левая башня](docs/qwen_two_tower.md), [правая башня](docs/track_tower_transformer.md). [Устройство текущего Colab runner](docs/colab_training.md).

## Установка

Python **3.12**. Для G4 нужен PyTorch с native Blackwell support:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
HNSWLIB_NO_NATIVE=1 pip install -r requirements.txt
```

Для CPU-проверок используйте CPU wheel PyTorch и `requirements-dev.txt`. Тесты используют маленький Qwen3 со случайными весами; pretrained модели для них не скачиваются. Полное обучение требует GPU. Ядро зафиксировано под текущий BF16-прогон; проверенный GPU — G4.

## Обучение и resume

Внешний датасет: `tracks_meta.parquet`, `track_embeddings.parquet` (MuQ128), `train.parquet`, `train_queries.parquet`, `train_qrels.parquet`. Первый запуск готовит fixed MiniLM признаки и токены; следующие переиспользуют кэш по fingerprint.

```bash
python tools/train_qwen.py --data-dir /data/music4all_crs \
  --artifacts /data/qwen_artifacts --epochs 3 --max-hours 12

python tools/train_qwen.py --data-dir /data/music4all_crs \
  --artifacts /data/qwen_artifacts --resume --epochs 5 --max-hours 24
```

У обычного CLI `--epochs` — **суммарный** предел целых эпох, `--max-hours` — **накопленный** активный лимит, включая прошлые запуски и подготовку. Перенос runs без budgets не обнуляет время из checkpoint. Resume требует совпадения данных, параметров и кода; восстанавливает optimizer/scaler, cursor и RNG. Обычный CLI использует исходную политику trainer; текущая Colab политика описана ниже.

Colab runner продолжает **ещё пять эпох** от сохранённого cursor: с 3,465 до 8,465 суммарно. Перед fit — 10 warmup + 100 измеренных обновлений с rollback весов, optimizer/RNG. Проверка качества — на 2 600 запросах каждые 1 000 шагов и на границах эпох. Резервные копии last/best отправляются в Drive раз в 1 800 секунд, также до fit и при завершении; small progress reports сохраняются отдельно. Early stopping: минимум одна дополнительная эпоха, 16 проверок без улучшения macro-nDCG@20 более чем на 0,0005. Лучшие веса сохраняются по фактическому максимуму.

## Инференс

Bundle: `artifacts/runs/<signature>/exports/<bundle_id>/`. Включает LoRA/проекции, item tower, tokenizer, preprocessing, каталог, history features, vectors и HNSW. Frozen Qwen загружается с закреплённой HF revision; файлы bundle проверяются по SHA256.

```bash
python tools/predict_qwen.py --bundle /data/bundle \
  --query 'Energetic rock for running' --history /data/past_history.json --top-k 200
```

История — JSON массив строковых `m4a_id` в хронологическом порядке с повторами. Последние 256 событий используются в query tower; вся переданная история — в `--discovery`. `--exact` переключает на весь каталог. Для similar_to: `--query-type similar_to --exclude-m4a-id ... --exclude-artist ...`.

```python
from recsys.two_tower_qwen.bundle import QwenRetriever

retriever = QwenRetriever('/data/bundle', device='cuda', cache_dir='/data/hf_cache')
vector = retriever.encode_query('Energetic rock for running', history=['track1', 'track2'])
raw_candidates = retriever.search('Energetic rock for running', history=['track1'], k=200)
```

`search()` возвращает raw ANN-кандидатов. Canonical-дедупликация и исключения добавляются CLI либо `CatalogScorer.clean_ranking`. Query вектор и индекс должны принадлежать одному bundle; старый EmbeddingGemma индекс с этим пространством несовместим.

## Оценка

Отдельно **exact** и **HNSW**: Recall@1/5/10/20/50/100/200, nDCG@20/100/200, MRR@200, Discovery и index overlap. При одной положительной canonical-цели Recall равен HitRate. Spotify aliases объединяются, similar_to исключает reference-трек и его исполнителя.

```bash
python jobs/colab_recall/prepare_inputs.py --data /data/music4all_crs \
  --bundle /data/bundle --output /data/holdout \
  --previous-validation /data/qwen_artifacts/runs/RUN/validation-step-55000.parquet

python jobs/colab_recall/evaluate.py --bundle /data/bundle \
  --inputs /data/holdout --output /data/recall \
  --credentials /private/drive_credentials.json --folder-id DRIVE_FOLDER_ID \
  --quota-project DRIVE_QUOTA_PROJECT --max-hours 3
```

Evaluator сохраняет в Drive пары exact/HNSW chunks до продвижения cursor. Веса и base cache в результаты не загружаются. Credentials не должны находиться в Git.

User holdout: **61 259** запросов; **2 600** участвовали в выборе checkpoint, **58 659** дополнительных показываются отдельно. Это внутренний holdout, пользователи между срезами могут пересекаться; это не независимый официальный test. Micro даёт вес каждому запросу, macro — одинаковый вес каждому из 13 типов. Diversity не заменяется самодельной формулой.

До текущего продолжения best checkpoint после 3,438 эпохи дал на полном внутреннем holdout Recall@100 **64,85%**, Recall@200 **73,05%**, micro-nDCG@20 **0,2672**. Новое обучение ещё продолжается. Эти числа не являются его итогом и не доказывают превосходство над BM25 на независимом test.

## Проверки

```bash
pip install -r requirements-dev.txt
python tools/verify_training_code.py
USE_TF=0 HF_HUB_OFFLINE=1 python -m pytest -q tests
```

Проверяются градиенты/frozen weights, padding/пустая история, aliases/Discovery, sampled softmax correction, resume/RNG rollback, bounded model loading и экспорт/загрузка. GPU smoke текущего прогона выполнен отдельно на RTX PRO 6000 Blackwell с BF16 и CUDA12.8.

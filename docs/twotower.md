# Двухбашенный кандген на semantic ID

Источник кандидатов `twotower`. Пользовательская башня — Qwen, ей на вход подаются история пользователя (каждый
трек — 4 токена кода RQ-VAE) и текст запроса. Трековая башня строит вектор трека по его коду. Обучение — softmax
по всему каталогу. Код: `recsys/twotower/`, источник: `recsys/retrieval/twotower.py`, параметры:
`configs/twotower.yaml`, коды треков: [semantic_ids.md](semantic_ids.md).

## Модель

```
вход:  Instruct: <instruction>\nQuery: History: [c1 c2 c3 c4] × L  Request: <запрос> <EOS>     (паддинг слева)
         текст — родной эмбеддер Qwen (заморожен)
         коды  — SemanticTokenEmbedding: P · центроид RQ-VAE + Δ + эмбеддинг позиции кода
                 (P: 32 → hidden; норма подогнана под обычные токены Qwen)
Qwen3-Embedding-0.6B (веса заморожены; на этапе B — LoRA на q, k, v, o)
скрытое состояние EOS (1024) → Linear → 256 → L2                                       = u

трек:  сумма строк таблицы по 4 кодам (старт — центроиды RQ-VAE, 32) → MLP → 256 → L2     = v
скор:  u · v / τ,  τ обучается (старт 0.05)
```

Номер трека везде — строка `semantic_codes.npy` (`tracks_meta` по возрастанию `m4a_id`).

**Лосс** (`loss.negatives`):
- `full` — cross-entropy по всем 64 016 трекам. Негативы — все треки каталога, кроме цели. Другие записи той же
  песни (9.2% треков: общий ISRC или одинаковые «артист + название») в негативы не идут, у них логит −∞.
- `in_batch` — негативы — цели других примеров батча; одна и та же цель или песня у двух примеров — не негатив;
  `logq: true` вычитает log частоты трека как цели.

**Этапы** (`train.stages`):
- **A** — Qwen заморожен, учатся таблицы кодов, проекция, трековая башня и τ.
- **B** — то же плюс LoRA на attention.

Новые случайные параметры сначала учатся отдельно, иначе их шумные градиенты испортят предобученную модель.

**История** (`input`): последние `history_len` = 64 уникальных трека. На train в половине примеров
(`history_dropout`) история обрезается до случайной длины 0..64. Поэтому модель работает при любой длине истории,
и ablation по длине считается на одной модели (`encode_twotower.py --history-len`).

**Данные**: только часть `index` запросов train (`twotower.train.part`, [recsys/data/parts.py](../recsys/data/parts.py)).
На частях `ranker_train` и `valid` учится и проверяется ранкер: если бы two-tower видел эти запросы, на них он находил
бы цель «по памяти», ранкер научился бы ему доверять, а на тесте этой памяти нет.

**Валидация**: 5% пользователей части `index` целиком. Метрика — recall@10 / 100 / 200 по всему каталогу; лучшая
модель выбирается по recall@100.

## Запуск

Один и тот же код работает на любом устройстве: cuda (DataSphere), mps (Mac), cpu. Устройство выбирается само,
`--device` задаёт его явно. От устройства зависят только точность (bf16 / fp16 — только на cuda, на mps и cpu
fp32) и реализация attention (на mps — `eager`: `sdpa` с паддингом слева даёт там NaN). Полное обучение Qwen 0.6B
рассчитано на GPU; локально — проверка кода на нескольких пользователях (`--n-users`, `train.max_steps`).

Нужны `torch`, `transformers>=4.51`, `peft` (`requirements.txt`, раздел двухбашенного кандгена) и
`artifacts/semantic_ids/` ([semantic_ids.md](semantic_ids.md)). Веса Qwen скачиваются с Hugging Face при первом
запуске.

```bash
# быстрая проверка: 300 пользователей, по 20 шагов на этап
python scripts/train_twotower.py --run-dir artifacts/twotower/smoke --n-users 300 --set twotower.train.max_steps=20

# основная модель -> artifacts/twotower/main: model.pt, last.pt, log.jsonl, metrics.json
python scripts/train_twotower.py --run-dir artifacts/twotower/main

# векторы треков и запросов частей ranker_train, valid, test (+ recall по всему каталогу, по типам) -> в ту же папку
python scripts/encode_twotower.py --run-dir artifacts/twotower/main
python scripts/encode_twotower.py --run-dir artifacts/twotower/main --parts submit   # для сабмита: весь test_public

# в отчёте кандгенов вместе с остальными (без GPU: читает векторы из папки)
python -m recsys candgen --sources +twotower --set candgen.sources.twotower.dir=artifacts/twotower/main --run tt
# в отборе: подобрать веса отбора с ним, затем обучить ранкер и оценить
python -m recsys tune-policy --sources +twotower --run tt
python -m recsys run --sources +twotower --config runs/tt/selection.yaml --run tt
```

Векторы запросов привязаны к частям датасета (`data.parts`): после изменения частей их нужно пересчитать.

В `inference.ipynb` (демо) — `candgen.sources.twotower.online: true`: запросы, которых нет в файлах, кодирует модель
(нужны torch и веса; на GPU быстро).

## Ablation

| что | команда |
|---|---|
| только этап A (без LoRA) | `--set 'twotower.train.stages=[{name: A, epochs: 1.3, lora: false, lr: 1.0e-3}]'` |
| in-batch / + logQ | `--set twotower.loss.negatives=in_batch` (`--set twotower.loss.logq=true`) |
| без маски дублей | `--set twotower.loss.mask_same_song=false` |
| случайные коды вместо RQ-VAE | `--set twotower.codes.source=random --set twotower.codes.init=random` |
| без старта с центроидов | `--set twotower.codes.init=random` |
| базовая Qwen | `--set twotower.model.name=Qwen/Qwen3-0.6B` |
| длина истории | без переобучения: `encode_twotower.py --history-len 0` (16, 32, ...) |

## Файлы прогона

| файл | что |
|---|---|
| `model.pt` / `last.pt` | обучаемые веса без весов Qwen: таблицы, проекции, трековая башня, τ, LoRA; конфиг `twotower` |
| `log.jsonl` | loss, τ, lr по шагам; метрики валидации |
| `metrics.json` | лучшая валидация и вся валидация после последнего шага, по типам запросов |
| `item_vectors.npy`, `item_ids.npy` | векторы треков и `m4a_id` |
| `query_vectors_<часть>.npz` | `query_id`, векторы запросов (float16) |
| `recall_<часть>.json` | recall@k по всему каталогу на части, по типам |

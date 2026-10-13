# Двухбашенный кандген на semantic ID

Источник кандидатов `twotower`. Пользовательская башня — Qwen, ей на вход подаются история пользователя (каждый
трек — 4 токена кода RQ-VAE и его теги) и текст запроса. Трековая башня строит вектор трека по коду и тегам. Обучение — softmax
по всему каталогу. Код: `recsys/twotower/`, кандген в пайплайне: `twotower` в `recsys/candgen/dense.py`, параметры:
`configs/twotower.yaml`, коды треков: [semantic_ids.md](semantic_ids.md).

## Модель

Две архитектуры (`twotower.arch`). Текущая — **v2**; v1 — первая модель (`artifacts/twotower/main`), её чекпоинты
без ключа `arch` загружаются как v1.

### v2: коды + теги, общие веса башен, короткий путь из истории

```
трек j:   коды RQ-VAE c1..c4                теги Last.fm и жанры (веса, сумма 1; recsys/twotower/tags.py)
            │ E_k[c_k] — SemanticTokenEmbedding   │ TagTable: взвешенная сумма эмбеддингов тегов (256)
            │ (общая таблица обеих башен)          │ (общая таблица обеих башен)
            ▼                                      ▼
  вход Qwen:  токен (j, k) = E_k[c_k] + Linear(теги j)          — 4 токена на трек, как в v1
  трек:       r_j = MLP(concat(LN(среднее E_k[c_k] → 256), LN(теги → 256)))
              v_j = L2(Linear(r_j))                                                   = кандидат

вход:  Instruct: …\nQuery: History: [трек 1: 4 токена] … [трек L]  Request: <запрос> <EOS>
Qwen3-Embedding-0.6B (заморожена; на этапе B — LoRA на q, k, v, o) → скрытое состояние EOS h (1024)

короткий путь:  a = softmax_j( q(h) · k(r_j) / √d  ∪  q(h) · пустой_слот )        — треки истории + «ничего»
                чтение = Σ_j a_j v_j + a_пусто · w_пусто                           — сами векторы кандидатов v_j
u = L2( Linear(h) + α · чтение )
скор = u · v / τ
```

- **Общие веса:** таблица кодов и таблица тегов одни и те же на входе Qwen и в векторе кандидата; трек в истории и
  тот же трек-кандидат описываются одними параметрами.
- **Короткий путь:** трек из истории, подходящий под запрос (большой вес a_j), сдвигает u прямо к своему v_j, не
  через слои Qwen. В v1 трек, видный модели во входе, находился в top-200 только в 41% случаев
  (`analyze_twotower.py`). Пустой слот — «ничего из истории»: запрос нового, пустая история. Его средний вес
  печатается в логе («пусто») и по типам запросов в `metrics.json` (`null`).
- **Без ID трека:** вектор трека — только из содержания (коды и теги), поэтому редкие треки не получают шумных
  векторов.

### v1 (первая модель)

```
вход:  те же 4 токена кода на трек (SemanticTokenEmbedding); u = L2(Linear(h))
трек:  своя таблица кодов (старт — центроиды RQ-VAE, 32) → MLP → 256 → L2                     = v
```

Результат v1 (1.3 эпохи): recall@200 по всему каталогу — 0.351 на валидации и 0.353 на test_public.

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

**Данные**: весь train (`twotower.train.part: train`). Поэтому в пайплайне twotower участвует в отборе кандидатов,
но не в признаках ранкера (`candgen.sources.twotower.features: false`): на ranker_train и valid он находит цель
«по памяти», и CatBoost научился бы ему доверять, а на тесте этой памяти нет. Учить на части `index`
(`--set twotower.train.part=index`) — тогда его признаки можно включить.

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

# основная модель -> artifacts/twotower/v2: model.pt, last.pt, log.jsonl, metrics.json
python scripts/train_twotower.py --run-dir artifacts/twotower/v2

# векторы треков и запросов частей ranker_train, valid, test (+ recall по всему каталогу, по типам) -> в ту же папку
python scripts/encode_twotower.py --run-dir artifacts/twotower/v2   # части ranker_train, valid, test, submit (~40 мин на A100)

# в отчёте кандгенов вместе с остальными (без GPU: читает векторы из папки)
python -m recsys candgen --sources +twotower --set candgen.sources.twotower.dir=artifacts/twotower/v2 --run tt
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
| v2 без тегов | `--set twotower.tags.enabled=false` |
| v2 без короткого пути | `--set twotower.shortcut.enabled=false` |
| v1 (коды, раздельные таблицы) | `--set twotower.arch=v1` |
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

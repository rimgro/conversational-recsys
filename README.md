# Разговорная рекомендация музыки

Датасет **Music4All-CRS** ([docs/dataset.md](docs/dataset.md)): каталог из 64k треков, история прослушиваний Last.fm и
текстовый профиль пользователя; на каждый трек, который пользователь послушал в целевом месяце, есть синтетический
запрос на английском. Задача: по запросу, истории и профилю найти этот трек (nDCG@20) и ответить текстом;
сабмит — `query_id`, `top20`, `response`.

```
запрос -> разбор -> кандгены (каждый отдельно) -> отбор 150 (взвешенный RRF) -> признаки -> CatBoost -> ответ
```

Один и тот же код считает метрики, обучает ранкер, делает сабмит и отвечает на реплику диалога (`Pipeline.run`).
Бейзлайн — решение team1 (ветка `team_1_baseline`, оригинальные ноутбуки — [reference/team1/](reference/team1/README.md)),
перенесённое в пакет: те же кандгены, отбор и признаки, но **без `query_type`** — тип запроса на тесте использовать
нельзя, поэтому отбор одинаков для всех запросов, а признака типа у ранкера нет.

## Запуск

```bash
pip install -r requirements.txt
pytest -q                                   # тесты: весь пайплайн на синтетике, без файлов датасета

python -m recsys run                        # обучить ранкер и посчитать метрики на valid и test -> runs/baseline/
python -m recsys submit                     # сабмит по всему test_public -> runs/baseline/submission_test_public.parquet

# лучший прогон (test nDCG@20 0.506): все 11 кандгенов team1 + три башни; нужны файлы башен в artifacts/ (ниже)
python -m recsys run --config configs/best.yaml --run best
python -m recsys submit --config configs/best.yaml --run best
```

В Google Colab — `pipeline.ipynb`: те же шаги ячейками, всё сохраняется на Drive, ответы на test_public и на
другой тестовый сплит (`--test-split test_private`).

Данные — Music4All-CRS в `music4all_crs/` (`data.dir`, в git не хранится). Индексы кандгенов собираются при первом
запуске (несколько минут) и кэшируются в `cache/`, как и выдача каждого кандгена: повторный прогон с другим ранкером
или весами отбора кандгены не пересчитывает.

| Команда `python -m recsys …` | Что делает |
|---|---|
| `candgen` | кандгены и отбор без ранкера: отчёт по каждому кандгену (быстро) |
| `tune-policy` | подобрать веса отбора на valid (после добавления кандгенов) |
| `train` | обучить ранкер: `ranker_train`, early stopping по `valid` |
| `evaluate` | метрики всех шагов на `--parts` (по умолчанию valid, test) |
| `run` | `train` + `evaluate` |
| `submit` | сабмит по всем запросам test_public |
| `build-index` | собрать индексы заранее |

Опции: `--run NAME` (папка `runs/NAME/`), `--sources +hnsw,-tfidf` (включить / выключить кандгены; без знаков —
только эти), `--set a.b=c` (любой параметр конфига), `--config extra.yaml`, `--parts valid,test`, `--synthetic`.

```bash
python -m recsys candgen --sources +card_bm25,+relisten,+audio_seed --run ours   # что дают наши кандгены
python -m recsys run --set ranker.catboost.depth=8 --run depth8
python -m recsys evaluate --set ranker.type=stub --run stub                       # без ранкера: порядок отбора
```

## Онлайн-инференс (`recsys/inference.py`)

Один запрос в формате датасета — строка `*_queries` и строка пользователя из `train` / `test_public` — проходит весь
пайплайн прогона (разбор → 14 кандгенов → отбор → признаки → CatBoost → ответ) и возвращает `query_id`, `top20`,
`response`. Всё грузится один раз при старте (~30 с без башен), запрос — один проход `Pipeline.run`.

```bash
# 1. собрать папку для сервера из прогона (что не нашлось — список в конце; повторный запуск докопирует)
python -m recsys.inference export --run lazy3_best --out serving --with-qwen \
  --model twotower=<v2/model.pt> --model twotower_lazy=<lazy_e0/model.pt> \
  --model item_text=<item_text/qwen3emb.npz> --model twotower_pavel=<экспорт Павла step-104000>
# 2. на сервере
python -m recsys.inference serve --bundle serving --port 8000          # POST /recommend, GET /health
python -m recsys.inference predict --bundle serving --queries q.parquet --users u.parquet --out answers.parquet
python -m recsys.inference check --bundle serving --queries music4all_crs/test_public_queries.parquet \
  --users music4all_crs/test_public.parquet --reference submission_test_public.parquet -n 200   # задержка, сверка
python -m recsys.inference chat --bundle serving --history britpop    # консоль: история из examples/histories.yaml,
                                                                       # запросы — с клавиатуры
```

```json
POST /recommend
{"query": {"query_id": "78ff70df6ca7c6df", "query": "90s british pop"},
 "user":  {"user_id": 35511, "history": ["tkLb1MuW7uxdZ9CW", "..."], "history_ts": [1247864903, "..."],
           "user_profile": "Listening since 2009; ...", "user_demographics": {"age": 21, "gender": "m", "country": "UK"},
           "split": "test_public"}}
-> {"query_id": "78ff70df6ca7c6df", "top20": ["...", "..."], "response": "I recommend ...", "timings": {"total": 0.08, "...": 0}}
```

`history` — все прослушивания по порядку, с повторами, как в датасете. Граница истории — по `split` (окно из
`data.windows`, как при обучении), иначе `cutoff` (unix), иначе сразу после последнего прослушивания. `query_type`
не нужен. Можно и списком запросов в одном POST.

Папка `serving/` (из `lazy3_best`; без моделей башен ~0.6 ГБ):

| что | откуда |
|---|---|
| `config.yaml`, `manifest.json` | конфиг прогона с путями внутри папки, башни `online: true`; что и откуда собрано |
| `data/tracks_meta.parquet` | датасет (целиком: md5 файла — ключ кэшей каталога и индексов) |
| `cache/catalog_*`, `cache/indexes/` | каталог и 5 текстовых индексов (собраны на части index) |
| `track_docs.parquet` | документы треков с `query_document` — на сервере нет train, из которого он строится |
| `ranker/` | CatBoost прогона (`model.cbm`, `features.json`) |
| `semantic_ids/` | коды RQ-VAE (`artifacts/semantic_ids`) — для башен v2 и lazy |
| `towers/twotower/`, `towers/twotower_lazy/` | `item_vectors.npy`, `item_ids.npy` + `model.pt` (с Drive, `twotower.ipynb` / `lazy.ipynb`) |
| `towers/item_text/qwen3emb.npz` | векторы названий треков для lazy (`scripts/encode_item_text.py`) |
| `towers/twotower_pavel/model/` | экспорт Павла (`config.json`, `tokenizer/`, `query_trainable.pt`, `history_features.npz`, `catalog.json`, `track_vectors.npy`) |
| `hf/` (`--with-qwen`) | Qwen3-Embedding-0.6B, база всех трёх башен (2.3 ГБ); без него — скачается с Hugging Face |

Сервер: `requirements.txt` + `torch`, `transformers>=4.51`, `peft`. Без башен пайплайн — ~22 мс на запрос (p50,
p95 ~32 мс, CPU). Башни — три прохода Qwen3-0.6B: на CPU ~1.5 с на запрос, на GPU десятки мс — нужен GPU
(`candgen.sources.<башня>.device`, по умолчанию cuda, если есть, иначе cpu; не mps: на GPU Mac векторы
расходятся с посчитанными на cuda — косинус ~0.5, на cpu 1.0000). Сверка без моделей: `export --lookup` кладёт
готовые векторы и списки запросов датасета вместо моделей — на `test_public` выдача и ответ совпадают с сабмитом
`lazy3_best` на 100% запросов. С моделями совпадение будет неполным: другие точность (fp32 / bf16) и поиск
(у Павла — HNSW), а для `similar_to` готовые списки Павла не содержат образца и его артиста — их убирали по qrels;
онлайн этого знания нет.

## Отчёт: метрики по шагам

`runs/<прогон>/<часть>/report.md` — на одних и тех же запросах:

| строка | что это |
|---|---|
| `<кандген>` | список кандгена как есть: recall@20 … @1000, recall (цель где угодно), only_this (нашёл только он), median_rank |
| `union` | все кандгены вместе — потолок для отбора |
| `selection` | 150 отобранных (вход ранкера); их nDCG@20 — выдача без ранкера (stub) |
| `ranker` | итоговая выдача |

Плюс nDCG@20 и recall по типам запросов (тип × список), какой фильтр выкинул цель, и `lists.csv`, `*_by_type.csv`,
`per_list.parquet` (запрос × список: место цели). Метрика как в датасете: nDCG@20 с одной целью, для `similar_to`
из выдачи убираются трек-образец и его артист.

## Валидация без утечки

Обучаемые кандгены (`query_tfidf` и `query_document` в BM25 — тексты прошлых запросов к треку) и ранкер
учатся на разных запросах. Иначе на своих же запросах кандген находит цель «по памяти», ранкер учится ему доверять,
а на тесте этой памяти нет. Поэтому train делится по тексту запроса (md5, `recsys/data/parts.py`):

| часть | запросы | для чего |
|---|---|---|
| `index` | train, бакеты 0–5 (без текстов из test_public) | обучение кандгенов |
| `ranker_train` | train, бакеты 6–8, 30k | обучение ранкера |
| `valid` | train, бакет 9, 4k | подбор весов отбора и параметров, early stopping |
| `test` | test_public, 10k | только итоговая оценка — по ней ничего не подбираем |
| `submit` | test_public, все | сабмит |

Перефразировки одного текста всегда в одной части; пересечение частей проверяется при загрузке.
Башни (`twotower`, `twotower_pavel`, `twotower_lazy`) учились на всём train, поэтому по умолчанию в признаки ранкера не
идут (`features: false`). В лучшем прогоне ([configs/best.yaml](configs/best.yaml)) идут: test башни не видели, и на нём
так лучше (+0.03–0.04 nDCG@20, [docs/experiments.md](docs/experiments.md)); valid при этом завышен.

## Конфиг ([configs/default.yaml](configs/default.yaml))

- `data` — папка датасета, части и их размеры;
- `candgen.sources.<имя>` — `enabled`, `top_k` и параметры каждого кандгена;
- `selection` — `limit` (150), `rrf_k` и `weights` — веса кандгенов во взвешенном RRF (одни для всех запросов),
  фильтры (`banned`, `exclude_artist`, `new_artist`, `exclude_tag`; по умолчанию выключены);
- `features.groups` — группы признаков; `ranker` — тип и параметры CatBoost;
- `summarizer`, `explainer`, `llm` — разбор запроса и текст ответа (шаблон или LLM).

## Кандгены (`recsys/candgen/`)

| имя | как ищет | по умолчанию |
|---|---|---|
| `bm25`, `metadata_bm25` | BM25 (bm25s) по документу трека (метаданные, текст песни, тексты прошлых запросов) / только метаданным | вкл |
| `tfidf`, `query_tfidf`, `metadata_char` | TF-IDF слов / по прошлым запросам к треку / символьных n-грамм (точные названия) | вкл |
| `history`, `history_recent`, `history_frequency` | свои треки: по прослушиваниям с затуханием 180 / 30 дней / по числу | вкл |
| `favorite_artists` | треки любимых артистов | вкл |
| `audio_taste`, `audio_recent` | MuQ: похожие на вкус / на недавнее | вкл |
| `card_bm25` | наш BM25 по карточке трека (поля с весами) по разбору запроса | выкл |
| `relisten` | своя история + совпадение карточки с запросом | выкл |
| `audio_seed` | MuQ: похожие на образец «like X by Y» (similar_to) | выкл |
| `hnsw` | EmbeddingGemma-2: запрос против векторов треков индекса hnsw-v1 (векторы — `scripts/encode_hnsw.py`) | выкл |
| `twotower` | двухбашенный: история (semantic ID) + запрос ([docs/twotower.md](docs/twotower.md)) | выкл |
| `twotower_lazy` | lazy-башня: энкодер истории + Qwen по запросу с cross-attention ([docs/lazy.md](docs/lazy.md)); векторы те же, что у `twotower`, папка `artifacts/twotower/lazy_e0` | выкл |
| `twotower_pavel` | Qwen history-aware two-tower (ветка codex/qwen-history-candgen): готовые top-200 по query_id (`recsys/candgen/precomputed.py`) | выкл |

Первые 11 — кандгены team1. Новый кандген в отбор попадает, только если у него есть вес в `selection.weights`: после
включения — `python -m recsys tune-policy --sources +<имя> --run X`, затем `run` с `--config runs/X/selection.yaml`.
Без этого он остаётся признаком ранкера (как `audio_taste` и `metadata_bm25` у team1).

**Новый кандген**: класс-наследник `Candgen` с `@register("имя", "вид")` и методом `retrieve(batch) -> [Cands]`
в `recsys/candgen/`, параметры — в `candgen.sources.<имя>`. Признаки `<имя>_source_rank` и `<имя>_source_hit` у
ранкера появятся сами.

### HNSW и two-tower

Оба — точный поиск по заранее посчитанным векторам (`recsys/candgen/dense.py`), без GPU на инференсе:

```bash
python scripts/make_embed.py                          # один раз: пакеты и доступ к EmbeddingGemma-2 (HF_TOKEN)
python scripts/encode_hnsw.py                         # векторы треков (индекс hnsw-v1) и запросов частей -> artifacts/hnsw
python -m recsys candgen --sources +hnsw --run hnsw   # что он даёт

python scripts/train_twotower.py --resume --run-dir artifacts/twotower/v2   # A100, ~6 ч; учится на всём train
python scripts/encode_twotower.py --run-dir artifacts/twotower/v2            # векторы частей ranker_train, valid, test, submit
python -m recsys tune-policy --sources +twotower --run tt                    # вес twotower в отборе
python -m recsys run --sources +twotower --config runs/tt/selection.yaml --run tt
```

Файлы трёх башен для [configs/best.yaml](configs/best.yaml):

| кандген | папка | что там | откуда |
|---|---|---|---|
| `twotower` | `artifacts/twotower/v2` | `item_vectors.npy`, `item_ids.npy`, `query_vectors_*.npz` | `twotower.ipynb` / `encode_twotower.py` |
| `twotower_lazy` | `artifacts/twotower/lazy_e0` | то же | `lazy.ipynb` / `encode_twotower.py` |
| `twotower_pavel` | `artifacts/twotower_pavel` | `train_top200.parquet`, `test_public_top200.parquet` | считаются вне репозитория |

Векторы запросов нужны для частей ranker_train, valid, test и submit (или файлы по сплитам: train, test_public).
Другой тестовый сплит (test_private): векторы его запросов — `encode_twotower.py --parts submit --set
data.parts.test_split=test_private` для обеих векторных башен; готовых списков `twotower_pavel` для него нет, пока
их не посчитают вне репозитория (без них прогон остановится; ранкер обучен с этой башней, поэтому выключить её нельзя).

Обучение и векторы — `twotower.ipynb` (`lazy.ipynb`) в Colab. Нет векторов для запросов части —
прогон останавливается с подсказкой, а не идёт молча без него.

Сервис HNSW пайплайн не вызывает (ищем по тем же векторам локально); `online: true` — новые реплики диалога
кодирует модель.

## Структура

```
configs/default.yaml        все параметры; best.yaml (лучший прогон), twotower.yaml, lazy.yaml, gemma.yaml — поверх default
recsys/
  __main__.py               точка запуска: python -m recsys <команда>
  run.py                    этапы прогона и отчёт; tune.py — подбор весов отбора
  pipeline.py               пайплайн: пачка запросов и одна реплика диалога (Pipeline.run -> Response)
  inference.py              онлайн-инференс: папка для сервера (export), HTTP (serve), файлы (predict), check
  resources.py              общее для шагов: данные, индексы, разбор запроса, состояния пользователей
  data/                     dataset.py (данные по конфигу), parts.py (части без утечки), users.py (история на момент
                            запроса), crs.py (чтение файлов), catalog.py, history.py (профиль), synthetic.py
  candgen/                  кандгены: text.py, card.py, history.py, audio.py, dense.py; cache.py — кэш выдачи
  selection.py              отбор: объединение, фильтры, взвешенный RRF
  features.py               признаки ранкера по группам
  ranker.py                 CatBoost и stub
  metrics.py                метрики по шагам
  dialog.py, llm.py         разбор запроса (правила / LLM)
  explain.py                ответ: шаблон / LLM
  schemas.py                Request, DialogSummary, Context, RankedTrack, Response
  twotower/, semantic_ids.py  двухбашенный кандген и semantic ID
scripts/                    GPU и подготовка: encode_hnsw, make_embed, train/encode/show/analyze_twotower,
                            check_semantic_ids
pipeline.ipynb              Colab: обучение ранкера, отчёт, ответы на test_public и другой сплит
twotower.ipynb              обучение двухбашенного кандгена на GPU (DataSphere / Colab)
reference/team1/            ноутбуки и артефакты бейзлайна team1 (эталон чисел)
docs/                       датасет, API сервисов, semantic ID, two-tower; experiments.md — результаты экспериментов
jobs/evaluate.yaml          python -m recsys run в DataSphere Jobs
llm_gemma_service/          Gemma-сервис для текста ответа (configs/gemma.yaml)
```

## Что важно в данных

- **Одна цель на запрос**, 13 типов (`query_type`; **модели его использовать нельзя** — это проверяется, он только
  для разбивки метрик): 11 к прослушанным трекам и два
  discovery — `novelty` (новый артист) и `similar_to` («like X by Y but from other artists»).
- **Большинство целей — повторные прослушивания**: кандгены истории — главный источник.
- **Запросы и теги на английском**, названия и имена — на языке оригинала.

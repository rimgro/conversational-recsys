# Разговорная рекомендация музыки

Датасет **Music4All-CRS** ([docs/dataset.md](docs/dataset.md)): каталог из 64k треков, история прослушиваний Last.fm и
текстовый профиль пользователя; на каждый трек, который пользователь послушал в целевом месяце, есть синтетический
запрос на английском. Задача: по запросу, истории и профилю найти этот трек (nDCG@20) и ответить текстом;
сабмит — `query_id`, `top20`, `response`.

```
запрос -> разбор -> кандгены (каждый отдельно) -> отбор 150 (RRF по типу запроса) -> признаки -> CatBoost -> ответ
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
Two-tower учится на всём train, поэтому в признаки ранкера не идёт (`candgen.sources.twotower.features: false`).

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

Обучение и векторы — `twotower.ipynb` в Colab. Two-tower учится на всём train, поэтому его скоры в признаки
CatBoost не идут (`features: false`): он только помогает отобрать 150 кандидатов. Нет векторов для запросов части —
прогон останавливается с подсказкой, а не идёт молча без него.

Сервис HNSW пайплайн не вызывает (ищем по тем же векторам локально); `online: true` — новые реплики диалога
кодирует модель.

## Структура

```
configs/default.yaml        все параметры; twotower.yaml, gemma.yaml — поверх default
recsys/
  __main__.py               точка запуска: python -m recsys <команда>
  run.py                    этапы прогона и отчёт; tune.py — подбор весов отбора
  pipeline.py               пайплайн: пачка запросов и одна реплика диалога (Pipeline.run -> Response)
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

# Разговорная рекомендация музыки

Датасет **Music4All-CRS** (описание: [docs/dataset.md](docs/dataset.md)): каталог из 64k треков, история
прослушиваний Last.fm и текстовый профиль пользователя; на каждый трек, который пользователь послушал
в целевом месяце, есть синтетический запрос на английском. Задача: по запросу, истории и профилю найти этот трек
(nDCG@20) и ответить текстом; сабмит — `query_id`, `top20`, `response`.

Схема: `scheme.png`. Код — в пакете `recsys/`, ноутбуки только запускают его:

| Ноутбук | Для чего |
|---|---|
| `inference.ipynb` | запрос пользователя → ответ и треки; следующая реплика |
| `experiments.ipynb` | почему такая выдача: разбор запроса по шагам, как правила понимают настоящие запросы, сравнение вариантов конфига, LLM |
| `examples/candgen.ipynb` | проверка сервиса HNSW руками (и старого сервиса BM25 — пайплайн его не использует) |

## Запуск

```bash
pip install -r requirements.txt
pytest -q                 # тесты, пайплайн целиком на синтетике
jupyter lab inference.ipynb   # или открыть в DataSphere
```

Вектор запроса для HNSW строим сами (EmbeddingGemma-2, GPU если есть, иначе CPU): один раз на машину
`python scripts/make_embed.py` — проверит пакеты, скачает веса, построит вектор и сверит его с индексом на сервере.

BM25 локальный: индекс карточек треков собирается из `tracks_meta` за ~20 с и кэшируется в `cache/` — один раз
`python scripts/make_index.py` (или сам при первом запуске). Сервис HNSW включён по умолчанию: задать `HNSW_URL`, `HNSW_API_KEY`
(файл `.env` в корне — он не в git — или секреты DataSphere). Без него: `USE_CANDGEN = False` в ноутбуке или `--offline`
у скриптов (остаются `relisten`, `audio`, `bm25`).

Данные — Music4All-CRS в папке `music4all_crs/` (`data.crs.dir`, в git не хранится): `tracks_meta.parquet`,
`train.parquet` / `test_public.parquet` (пользователи), `*_queries.parquet` (запросы), `*_qrels.parquet` (ответы).
Синтетика в формате датасета (`data.source: synthetic`) нужна только тестам: они не зависят от файлов.
В DataSphere: открыть ноутбук из корня репозитория. Описание выдачи от LLM-сервиса проекта — `USE_LLM_SERVICE = True`
в `inference.ipynb` (подключает `configs/gemma.yaml`). Код запуска находится в [llm_gemma_service/](llm_gemma_service/README.md).
Первичная настройка — `llm_gemma_service/gemma_datasphere_service.ipynb`; если сервер уже работает на той же ВМ,
он используется повторно. Веса и процесс остаются вне репозитория, обновления pipeline не перезапускают LLM.
Локально сервиса нет, и ответ пишет шаблон. Своя модель в ноутбуке — `llm.type: local` (GPU и `transformers`).

## Сервисы-кандгены

Сервисы разрабатываются и разворачиваются в своих ветках, здесь только клиент к ним (`recsys/retrieval/remote.py`)
и контракт [docs/candgen_api.md](docs/candgen_api.md):

| Ветка | Что | Сейчас |
|---|---|---|
| `bm25-cards` | сервис BM25 по карточке трека | развёрнут, но пайплайн его не использует: BM25 локальный (`local_index.py`) |
| `dev/hnsw`, релиз `hnsw-v1` | текстовый семантический поиск (EmbeddingGemma + LanceDB, `POST /hnsw/search`) | развёрнут на сервере |
| `dev-ranker` | ранкер LightGBM | в разработке |

- Источник `hnsw` в `configs/default.yaml` — это он. Адрес и ключ — `HNSW_URL`, `HNSW_API_KEY` (файл `.env` или секреты
  DataSphere).
- Если сервис недоступен, его источники возвращают пустой список с предупреждением, остальные работают.

## Валидация (`scripts/evaluate.py`)

Метрики пайплайна на сплите датасета: роль валидации играет `test_public` (12k пользователей, 273k запросов),
`test_private` скрыт. Метрика как в датасете — nDCG@20 с одной целью (для `similar_to` из выдачи убираются трек-образец
и его артист), плюс hit@20 и MRR — по типам запросов и по новым/знакомым трекам; recall каждого источника кандидатов
(`recall@<источник>`) и всех вместе (`recall@fused`) — качество кандгенов.

```bash
python scripts/evaluate.py --n-users 1000                   # 1000 пользователей, HNSW с сервера
python scripts/evaluate.py --n-users all                    # весь сплит
python scripts/evaluate.py --n-users 1000 --offline         # без HNSW (relisten, audio, bm25), ~1 мин
python scripts/evaluate.py --set ranker.type=heuristic      # любой параметр конфига
```

Результат — `outputs/<время>_<сплит>/`: `metrics.json` (метрики, конфиг, git-коммит, версии индексов сервисов,
ошибки сервисов), `submission.parquet` (сабмит), `by_query_type.csv`, `by_is_new.csv`, `per_request.csv`
(дописывается по ходу), `config.yaml`.
Прогон не начнётся, если сервис недоступен (код выхода 2).

Скорость: ~130 запросов/с без HNSW, ~2 запроса/с с HNSW (он отвечает ~0.5 с).
Короткие прогоны — из ноутбука (`!python scripts/evaluate.py ...`) или локально; полный — в DataSphere Jobs без открытого
ноутбука: `datasphere project job execute -p <id проекта> -c jobs/evaluate.yaml` (что подготовить — в начале файла).

### Кандгены отдельно (`scripts/evaluate_candgen.py`)

Шаги 1–3 без ранкера и описания (локально ~170 запросов/с): каждый источник кандидатов и три этапа слияния —
`rrf` (все источники до фильтров), `filtered` (после фильтров и ограничений запроса), `fused` (первые `fusion.top_n`,
вход ранкера; его nDCG@20 — это выдача stub-ранкера).

```bash
python scripts/evaluate_candgen.py --n-users 1000                     # все источники
python scripts/evaluate_candgen.py --n-users 1000 --offline           # без сервисов
python scripts/evaluate_candgen.py --sources bm25,audio               # только эти источники
python scripts/evaluate_candgen.py --split train --set fusion.exclude_top_tags=5   # подбирать параметры — на train
```

Результат — `outputs/<время>_<сплит>_candgen/`: `candgen.csv` (recall, recall@20/50/100/200, nDCG@20, only_this —
цель нашёл только он, median_rank, сколько кандидатов), то же по типам запросов, `recall_by_query_type.csv` и
`ndcg_by_query_type.csv` (тип запроса × список), `filter_losses.csv` (какой фильтр выкинул найденную цель).

## Что важно в данных

- **Одна цель на запрос**, 13 типов (`query_type`): 11 к прослушанным трекам (exact, lyrics_recall, lyrics_theme,
  genre, mood, situation, era_region, audio_attributes, complex, negative_constraint, vague_recall) и два discovery:
  `novelty` (новый артист, вне привычного вкуса) и `similar_to` («like X by Y but from other artists»).
  Метрики считаем по типам (`metrics_by`).
- **Большинство целей — повторные прослушивания** (`is_new = False`). Поэтому прослушанное по умолчанию не выкидываем
  (`fusion.exclude_listened: false`), а свою историю ищем отдельным источником `relisten`. Для `similar_to` и `novelty`
  разбор запроса ставит флаги «только новые треки» / «только новые артисты».
- **Запросы и теги на английском**, названия и имена — на языке оригинала.

## Пайплайн (`recsys/pipeline.py`)

| Шаг | Модуль | Сейчас |
|---|---|---|
| 1. разбор запроса → `DialogSummary` | `dialog.py` | правила (`rule`), `llm` с фолбэком на правила |
| 2. кандидаты | `retrieval/sources.py` | см. ниже |
| 3. RRF + фильтры + ограничения запроса + дедуп | `fusion.py` | настоящий |
| 4. ранкер | `ranking.py` | `stub` (порядок RRF), `heuristic`, `api` |
| 5. описание | `explain.py` | `stub` (приветствие + список), `llm` |

Фильтры и ограничения (шаг 3, `fusion.py`):

- **жёсткие фильтры** (`Filters`): показанное и скипнутое; прослушанное — только для similar_to / novelty;
  артисты из `exclude_artists`; для novelty — все артисты истории; `exclude_tags` («no pop») — только если тег среди
  первых `fusion.exclude_top_tags` тегов трека по весу.
- **ограничения запроса** (`fusion.constraints`): год / десятилетие, тональность, bpm, «high/low energy», instrumental,
  пол вокала, явный язык (`text.parse_constraints`). Синтетические запросы пишутся по мете цели, и цель им удовлетворяет
  в ~97% случаев (проверено на train), поэтому нарушители уходят вниз списка (штраф к RRF), но не выкидываются.
  Страна артиста — нет: цель ей удовлетворяет в ~80%. Признак `n_violations` есть у ранкера.

Источники кандидатов — четыре, у каждого своя задача:

| Источник | Как ищет | Зачем |
|---|---|---|
| `relisten` | треки из истории пользователя: совпадение с запросом (тот же BM25 карточки, только по истории) + сколько и как недавно слушал | 93% целей — повторные прослушивания, другие источники их почти не находят |
| `audio` | MuQ-эмбеддинги: ближайшие к треку-образцу «like X by Y» или к центру вкуса | similar_to (новые треки, похожие по звучанию) |
| `bm25` | локальный BM25 по карточке трека: теги, жанры, артист, название, год + текст песни + описания (поля с весами `bm25_index.weights`) | genre, era_region, complex, exact, lyrics |
| `hnsw` | сервис HNSW: вектор реплики (EmbeddingGemma строим сами, `scripts/make_embed.py`) | смысл запроса: vague_recall, lyrics_theme, mood |

Контракты между шагами: `recsys/schemas.py` (`Request`, `DialogSummary`, `Candidate`, `FusedCandidate`, `RankedTrack`, `Response`).

## Структура

```
inference.ipynb            инференс: запрос -> ответ
experiments.ipynb          эксперименты: разбор по шагам, сравнение вариантов
examples/candgen.ipynb     проверка сервиса HNSW руками
configs/default.yaml       все параметры и переключатели
docs/dataset.md            описание датасета
docs/candgen_api.md        API сервисов-кандгенов (контракт с нашей частью)
docs/semantic_ids.md       semantic ID треков: файлы, RQ-VAE по MuQ, что проверено
jobs/evaluate.yaml         scripts/evaluate.py в DataSphere Jobs
scripts/                   запускать из корня репозитория: python scripts/<скрипт>.py
  evaluate.py              валидация: метрики на test_public -> outputs/
  evaluate_candgen.py      метрики кандгенов: каждый источник и этапы слияния, без ранкера
  make_index.py            предподсчёт локального BM25-индекса карточек (cache/)
  make_embed.py            подготовка вектора запроса для HNSW: пакеты, веса EmbeddingGemma, проверка (GPU или CPU)
  check_semantic_ids.py    проверка semantic ID (artifacts/semantic_ids/): коллизии, порядок строк, дубли песен
recsys/
  schemas.py      контракты между шагами
  config.py       загрузка YAML + overrides (предупреждает об опечатках в ключах)
  pipeline.py     пять шагов схемы
  semantic_ids.py коды RQ-VAE треков, кодбуки, RQ-VAE в numpy (двухбашенный кандген)
  text.py         нормализация тегов, токенизация, эпохи / страны / языки в запросе
  llm.py          StubLLM, LocalLLM (transformers), extract_json
  dialog.py       шаг 1: RuleSummarizer (заглушка), LLMSummarizer, промпты
  retrieval/
    local_index.py  локальный BM25 по карточке трека (поля с весами), кэш индекса
    sources.py    шаг 2: relisten, audio и сборка источников из конфига
    remote.py     шаг 2: сервисы bm25 и hnsw по HTTP
  fusion.py       шаг 3: RRF, фильтры, ограничения запроса, веса источников
  ranking.py      шаг 4: признаки, StubRanker, HeuristicRanker, APIRanker
  explain.py      шаг 5: StubExplainer, LLMExplainer (ответ по-английски)
  eval.py         nDCG@20 и др., evaluate, metrics_by, compare_configs
  data/
    catalog.py    каталог треков + эмбеддинги
    crs.py        чтение Music4All-CRS: tracks_meta -> Catalog, users + queries + qrels -> Request на query_id
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

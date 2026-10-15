# Музыкальные рекомендации по запросу: инференс

Ветка `vps` — только то, что нужно, чтобы на сервере отвечать на запрос: история пользователя и текст запроса
(в формате датасета Music4All-CRS) → 20 треков (`m4a_id`) и текст ответа. Обучение, оценка, ноутбуки и
документация — в ветке `dev/baseline`.

Модель — прогон `final6` (test nDCG@20 0.5267 без бонуса за новизну; с бонусом λ = 2 — 0.505, Discovery@20 0.69):

```
запрос → разбор (желаемое / исключённое, «like X by Y», прозвища и опечатки артистов)
       → 6 кандгенов: bm25, history, башни v2, lazy и Павла (Qwen3-Embedding-0.6B), audio_seed (MuQ-соседи образца)
       → отбор 150 (RRF, фильтры исключений) → признаки → CatBoost → бонус за новизну → top-20 → ответ
```

## Что в архиве

```
recsys-vps/
  recsys/            код (python -m recsys.inference)
  serving/           модель: конфиг, ранкер, индексы, документы треков, векторы и веса трёх башен, Qwen (hf/)
    examples/        20 запросов test_public, их пользователи и эталонная выдача сабмита — для проверки
  examples/histories.yaml   истории для консоли (chat)
  setup.sh           установка и проверка
  requirements.txt   версии пакетов
  tests/             тесты кода (на синтетике, без модели)
  Dockerfile         образ (не проверялся — проверенный путь setup.sh)
```

Всё нужное внутри: интернет на сервере нужен только для `pip install`, веса Qwen лежат в `serving/hf/`.

## Сервер

- Python 3.9–3.12, Linux или macOS;
- память: **≥ 12 ГБ RAM** (три копии Qwen3-0.6B в fp32 на CPU ~7.5 ГБ + индексы и каталог);
- диск: ~3 ГБ модель + ~2 ГБ окружение;
- CPU: ~1.3 с на запрос (три прохода Qwen); с GPU (CUDA) — десятки мс, setup.sh поставит torch с CUDA сам,
  если есть `nvidia-smi`.

## Установка с чистого листа

```bash
tar -xzf recsys-vps-final6.tar.gz
cd recsys-vps
./setup.sh                     # другой Python: PYTHON=python3.11 ./setup.sh
```

`setup.sh` создаёт `.venv`, ставит пакеты, запускает тесты и `check`: 20 запросов из `serving/examples/` через
всю модель, задержка по шагам и совпадение с эталоном. Эталон — сабмит, где башни брали векторы, посчитанные на
GPU; здесь они кодируют запрос сами (fp32 на CPU), поэтому совпадение top-20 высокое, но не обязательно 100%.

## Запуск

```bash
.venv/bin/python -m recsys.inference serve --bundle serving --port 8000      # HTTP; старт ~1 мин
curl -s localhost:8000/health
curl -s localhost:8000/recommend -H 'Content-Type: application/json' -d '{
  "query": {"query_id": "q1", "query": "calm acoustic songs for a rainy evening, no rock"},
  "user":  {"history": ["tkLb1MuW7uxdZ9CW", "00mLjKFmaTv9MdaB"], "history_ts": [1581000000, 1581100000]}}'
```

Ответ: `{"query_id", "top20": [m4a_id × 20], "response": "...", "timings": {шаг: секунды}}`. Можно и списком
запросов в одном POST. Формат входа — строки датасета: `query` — строка `*_queries.parquet` (нужен `query`,
остальное по желанию), `user` — строка `train` / `test_public.parquet`: `history` (все прослушивания по порядку,
с повторами), `history_ts`, `user_profile`, `user_demographics`, `split`. `query_type` не нужен и не читается.
Граница истории — по `split` (окно датасета), иначе `cutoff` (unix), иначе сразу после последнего прослушивания.

Другие режимы:

```bash
# консоль: история из examples/histories.yaml, запросы с клавиатуры
.venv/bin/python -m recsys.inference chat --bundle serving --history britpop
# файл запросов -> файл ответов (как сабмит: query_id, top20, response)
.venv/bin/python -m recsys.inference predict --bundle serving --queries q.parquet --users u.parquet --out answers.parquet
```

Как сервис systemd (`/etc/systemd/system/recsys.service`, путь — свой):

```ini
[Service]
WorkingDirectory=/opt/recsys-vps
ExecStart=/opt/recsys-vps/.venv/bin/python -m recsys.inference serve --bundle serving --port 8000
Restart=on-failure
[Install]
WantedBy=multi-user.target
```

## Настройки

`serving/config.yaml` — конфиг прогона с путями внутри папки; любой ключ можно поменять при запуске:
`--set ranker.discovery.bonus=0` (без бонуса за новизну), `--set candgen.sources.twotower.device=cuda`.
Для живых запросов включены `summarizer.fuzzy_artists` (опечатки: «no nirnava») и `summarizer.resolve_entities`
(«no chili peppers» — по согласию BM25 и башен).

## Как собрать архив заново (на машине разработки, ветка dev/baseline)

```bash
python -m recsys.inference export --run final6 --out ../recsys-vps/serving --with-qwen \
  --set ranker.discovery.bonus=2 --set ranker.discovery.keep_top=3 \
  --set summarizer.fuzzy_artists=true --set summarizer.resolve_entities=true
```

Нужны прогон `runs/final6` (ранкер), `artifacts/` (векторы и веса башен, экспорт Павла в
`artifacts/twotower_pavel/bundle`, `artifacts/item_text`, `artifacts/semantic_ids`), датасет и кэш индексов.
Затем в `../recsys-vps`: примеры для проверки в `serving/examples/` и архив — см. `make_archive.sh`.

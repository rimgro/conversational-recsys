# Кандгены на VPS

BM25 и HNSW — два независимых сервиса на одной машине: свой процесс, свой venv, свой порт, свои индексы.
Наша часть в DataSphere ходит к ним по HTTP (`recsys/retrieval/remote.py`), API — [docs/candgen_api.md](../docs/candgen_api.md).

| Сервис | Порт | Код | systemd | Индексы |
|---|---|---|---|---|
| BM25 | 8001 | `bm25/` | `recsys-bm25` | `indexes/bm25/{genres,tags,title}` |
| HNSW | 8002 | `hnsw/` | `recsys-hnsw` | `indexes/hnsw/audio` |

Нужно: Linux с systemd, Python 3.10+ (`python3-venv`). Пути ниже — `/opt/recsys`; если другие, поправить их в `recsys-*.service`.

## Проверить локально

То же, что на VPS, но без systemd: оба сервиса отдельными процессами на 8001 и 8002 (команды как в юнитах).

```bash
pip install -r bm25/requirements.txt -r hnsw/requirements.txt       # Python 3.10+
sh deploy/run_local.sh data/tracks_meta.parquet                     # строит indexes/, если их нет; Ctrl+C — стоп
python deploy/check.py                                              # в другом терминале: health, поиск, общие id
```

Вместо `tracks_meta.parquet` можно передать папку с частями `tracks_meta-*.parquet` (так выложен полный датасет),
например `data/full`. Посмотреть запросы и ответы руками — `examples/candgen.ipynb`.

Дальше `inference.ipynb` с `USE_CANDGEN = True` (адреса по умолчанию — localhost:8001 и :8002).
Локально не проверяются только сами юниты systemd (пользователь, пути, автозапуск); на VPS их проверяет
`systemctl status` и тот же `check.py`.

## Первый запуск

```bash
sudo useradd -r -s /usr/sbin/nologin recsys
sudo mkdir -p /opt/recsys /etc/recsys && sudo chown recsys: /opt/recsys
sudo -u recsys git clone -b <ветка> <repo> /opt/recsys && cd /opt/recsys   # или скопировать репозиторий (scp/rsync)
sudo -u recsys mkdir -p data && sudo install -o recsys <tracks_meta.parquet> data/

# у каждого кандгена свой venv
sudo -u recsys python3 -m venv .venv-bm25 && sudo -u recsys .venv-bm25/bin/pip install --no-cache-dir -r bm25/requirements.txt
sudo -u recsys python3 -m venv .venv-hnsw && sudo -u recsys .venv-hnsw/bin/pip install --no-cache-dir -r hnsw/requirements.txt

# индексы
sudo -u recsys env BM25_PYTHON=.venv-bm25/bin/python HNSW_PYTHON=.venv-hnsw/bin/python \
    sh deploy/build_indexes.sh all data/tracks_meta.parquet indexes

sudo cp deploy/recsys-bm25.service deploy/recsys-hnsw.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now recsys-bm25 recsys-hnsw
sudo ufw allow 8001/tcp && sudo ufw allow 8002/tcp          # если включён ufw
```

Проверка — на VPS, со своего компьютера и из ячейки ноутбука в DataSphere (`!python deploy/check.py ...`):

```bash
python3 deploy/check.py http://<IP>:8001 http://<IP>:8002
```

Он проверяет health, поиск во всех индексах и что id из BM25 находятся в HNSW (индексы из одного `tracks_meta`).
Если с VPS работает, а из DataSphere нет — DataSphere не пускает наружу или порты закрыты файрволом.

В DataSphere: переменные `BM25_URL=http://<IP>:8001`, `HNSW_URL=http://<IP>:8002`, в ноутбуке `USE_CANDGEN = True`.

## Обновление одного сервиса

Остальные сервисы при этом не перезапускаются; пока сервис лежит (секунды), наша часть пропускает его кандидатов.

```bash
cd /opt/recsys && sudo -u recsys git pull
sudo -u recsys .venv-hnsw/bin/pip install --no-cache-dir -r hnsw/requirements.txt   # если менялись зависимости
sudo -u recsys env HNSW_PYTHON=.venv-hnsw/bin/python \
    sh deploy/build_indexes.sh hnsw data/tracks_meta.parquet indexes             # если менялся индекс
sudo systemctl restart recsys-hnsw
```

Для BM25 то же с `bm25`. Индексы читаются один раз при старте, поэтому после пересборки нужен `restart`.
Если меняется API, сначала свериться с разделом «Как менять» в `docs/candgen_api.md`.

## Ключи

Сейчас доступ без ключей и по HTTP: любой, кто знает IP и порт, может делать запросы (только чтение).
Включить ключ для сервиса:

```bash
echo 'BM25_API_KEY=<ключ>' | sudo tee /etc/recsys/bm25.env && sudo chmod 600 /etc/recsys/bm25.env
sudo systemctl restart recsys-bm25
```

(для HNSW — `HNSW_API_KEY` в `/etc/recsys/hnsw.env`), а в DataSphere задать тот же `BM25_API_KEY` / `HNSW_API_KEY` секретом.
Без HTTPS ключ идёт открытым текстом.

## Логи и память

```bash
journalctl -u recsys-bm25 -f                 # логи (загруженные индексы и их версии — при старте)
systemctl status recsys-bm25 recsys-hnsw     # строка Memory: сколько RAM держит сервис
```

Каждый сервис держит свои индексы в памяти; uvicorn запущен одним процессом, чтобы не дублировать их.

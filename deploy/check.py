"""Проверка кандгенов по адресам: health, поиск во всех индексах, общие id у BM25 и HNSW, задержка.

    python deploy/check.py                                          # локально (deploy/run_local.sh)
    python deploy/check.py http://<IP>:8001 http://<IP>:8002        # VPS, с ноутбука или из DataSphere
    !python deploy/check.py $BM25_URL $HNSW_URL                     # из ячейки ноутбука

Только стандартная библиотека. Ключи, если включены: BM25_API_KEY, HNSW_API_KEY. Код выхода 1 при ошибке.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

WORDS = {'genres': ['rock'], 'tags': ['rock'], 'title': ['the']}


def call(url, path, body=None, key=''):
    headers = {'Content-Type': 'application/json', **({'X-API-Key': key} if key else {})}
    req = urllib.request.Request(url.rstrip('/') + path, data=None if body is None else json.dumps(body).encode(),
                                 headers=headers, method='GET' if body is None else 'POST')
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'{path}: HTTP {e.code} {e.read().decode("utf8", "replace")[:200]}') from None
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f'{path}: {e}') from None
    return out, (time.perf_counter() - t0) * 1000


def main(bm25_url, hnsw_url):
    bkey, hkey = os.environ.get('BM25_API_KEY', ''), os.environ.get('HNSW_API_KEY', '')
    ok = True

    def step(name, fn):
        nonlocal ok
        try:
            print(f'ok    {name}: {fn()}')
        except Exception as e:  # noqa: BLE001 — печатаем любую ошибку и идём дальше
            ok = False
            print(f'FAIL  {name}: {e}')

    found = []

    def bm25_health():
        h, ms = call(bm25_url, '/health')
        return f"{ms:.0f} мс, индексы " + ', '.join(f"{n} ({i['n_items']} треков, {i['index_version']})"
                                                     for n, i in h['indexes'].items())

    def bm25_search(index):
        def run():
            r, ms = call(bm25_url, '/bm25/search', {'words': WORDS.get(index, ['rock']), 'k': 10, 'index': index}, bkey)
            found.extend(r['ids'])
            return f"{ms:.0f} мс, {len(r['ids'])} треков по {WORDS.get(index, ['rock'])}"
        return run

    def hnsw_health():
        h, ms = call(hnsw_url, '/hnsw/health')
        if h['status'] != 'ok':
            raise RuntimeError(f"status {h['status']}: нет индексов")
        return f"{ms:.0f} мс, индексы " + ', '.join(f"{n} ({i['n_items']} треков, dim {i['dim']}, {i['index_version']})"
                                                     for n, i in h['indexes'].items())

    def hnsw_search():
        if not found:
            raise RuntimeError('нет id из BM25, не с чем сравнить')
        r, ms = call(hnsw_url, '/hnsw/search', {'track_ids': found[:5], 'k': 10, 'exclude_ids': found[:5]}, hkey)
        if r['n_query_tracks'] == 0:
            raise RuntimeError('ни один id из BM25 не найден в HNSW: индексы собраны из разных tracks_meta?')
        return f"{ms:.0f} мс, {len(r['ids'])} похожих; id из BM25 найдено в HNSW: {r['n_query_tracks']} из {len(found[:5])}"

    step(f'BM25 {bm25_url} /health', bm25_health)
    for index in WORDS:
        step(f'BM25 поиск, индекс {index}', bm25_search(index))
    step(f'HNSW {hnsw_url} /hnsw/health', hnsw_health)
    step('HNSW поиск по трекам из BM25', hnsw_search)
    print('всё работает' if ok else 'есть ошибки')
    return 0 if ok else 1


if __name__ == '__main__':
    args = sys.argv[1:]
    bm25 = args[0] if args else os.environ.get('BM25_URL') or 'http://127.0.0.1:8001'
    hnsw = args[1] if len(args) > 1 else os.environ.get('HNSW_URL') or 'http://127.0.0.1:8002'
    sys.exit(main(bm25, hnsw))

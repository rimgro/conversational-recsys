# Lazy-кандген: энкодер истории + декодер по запросу

`twotower.arch: lazy` — та же двухбашенная схема и тот же лосс, что у v2 ([twotower.md](twotower.md)), но история
пользователя не стоит в промпте Qwen: её обрабатывает энкодер, а Qwen по тексту запроса смотрит на неё через
cross-attention. Код: `recsys/twotower/lazy.py`, конфиг: `configs/lazy.yaml`.

Вопрос эксперимента: сколько стоит и что даёт обработка истории. `encoder_layers = 0` — lazy (энкодера нет,
cross-attention сразу к эмбеддингам треков), 2, 4 … — слои энкодера; `mode: prompt` — decoder-only бейзлайн
(история в промпте, как v2).

## Модель (mode: encdec)

```
трек j (все 64 016 на каждом шаге):
  коды RQ-VAE: 4 таблицы × 32 (старт — центроиды) -> склейка 128 -> MLP 128→256→256 -> LN       256
  теги: взвешенная сумма строк таблицы 12 472 × 256 -> LN                                         256
  название «артист — название»: вектор Qwen3-Embedding 1024 (заранее) -> Linear 1024→256 -> LN   256
  склейка 768 -> MLP 768→512→256                                                         = r_j  256
  v_j = L2(Linear 256→256 (r_j))                                                         = кандидат

история: 256 треков (сумма exp(-давность/180 дн) по прослушиваниям; порядок — по последнему прослушиванию)
  токен = LN( Linear 256→1024 (r_j) + E_pos[место по свежести] + E_count[log2 прослушиваний] + E_age[log2 дней] )
                                                                                         (L × 1024)
1. ЭНКОДЕР: E слоёв (копии слоёв 0..E-1 Qwen, обучаются целиком), только история, attention в обе стороны
   -> память M = [пустой слот, выход энкодера]                                          ((1 + L) × 1024)
   E = 0: память — сами токены истории

2. ДЕКОДЕР: 28 слоёв Qwen (заморожены; LoRA на этапе B), только текст:
   Instruct: Given a user's listening history and a request, … \nQuery: <запрос> <EOS>  (~34 токена)
   перед каждым слоем l (cross_every = 1 -> 28 блоков):
     x = x + tanh(g_l) · CrossAttn_l(q: RMSNorm(x), k/v: RMSNorm(M))
     q / k / v / o — копия self-attention слоя l (16 голов q, 8 голов k/v, по 128), свои нормы, без RoPE;
     g_l = 0 на старте: декодер — в точности исходная Qwen
   затем сам слой l (self-attention по тексту, MLP)

u = L2(Linear 1024→256 (h_EOS)),  скор = u·v/τ,  softmax по 64 016 трекам (другие записи той же песни — не негативы)
```

- **Один трек — один токен** (в v2 — 4 токена кодов, поэтому там было 64 трека). Цель попадает в последние
  64 уникальных трека в 21.7% запросов train, в 256 по затуханию — в 45.6% (в истории вообще — 65.9%).
- **Пустой слот** в памяти — «ничего из истории»: пустая история, запрос нового; без него softmax по пустой
  памяти даёт NaN.
- **Порядок истории** — `E_pos` (в cross-attention RoPE нет; в энкодере RoPE слоёв Qwen тоже есть).
- **Shortcut v2 нет**: он обходит декодер и скрыл бы разницу между вариантами.
- `mode: prompt` совпадает с Qwen из transformers на `[история, текст]`, `encdec` при g = 0 — с Qwen на тексте
  (`tests/test_lazy.py`).

## Обучение

| | учится | lr (`train.stages`) |
|---|---|---|
| башня треков, вход истории, выход, τ, гейты, пустой слот | ~4.6M | `lr` (1e-3 / 5e-4) |
| энкодер (15.7M на слой) и блоки cross-attention (6.3M на блок, 176M на 28) — копии слоёв Qwen | 31M при E=2 + 176M | `lr_copied` 1e-4 |
| LoRA r=16 на q, k, v, o декодера | 4.6M | только этап B, `lr_lora` 2e-4 |
| Qwen3-Embedding-0.6B | — | заморожена |

В логе `гейт` — средний |tanh(g)|: насколько декодер уже смотрит на историю.

## FLOPs (Qwen3-0.6B, история 254 трека, текст 34 токена; GFLOP на запрос, forward)

| вариант | encoder | memory_kv | query | всего = пример на обучении | кэш k/v на пользователя |
|---|---|---|---|---|---|
| encdec, E = 0, cross каждый слой | 0 | 30 | 40 | **70** | 29 МБ |
| encdec, E = 2 | 17 | 30 | 40 | 87 | 29 МБ |
| encdec, E = 4 | 34 | 30 | 40 | 104 | 29 МБ |
| encdec, E = 0, cross каждый 4-й | 0 | 7.5 | 33 | 40 | 7 МБ |
| prompt (decoder-only) | 241 | 0 | 32 | **273** | 29 МБ |

`encoder` и `memory_kv` зависят только от пользователя: их можно считать офлайн и хранить, тогда онлайн — только
`query`. Decoder-only тоже может так (история перед текстом, attention causal) — с кэшем k/v всех слоёв.
Считает `lazy_flops`; при обучении — в лог и в `metrics.json` (`flops`).

## Запуск

```bash
python scripts/encode_item_text.py          # один раз: векторы названий -> artifacts/item_text/qwen3emb.npz
python scripts/train_twotower.py --config configs/lazy.yaml --run-dir artifacts/lazy/e0
python scripts/train_twotower.py --config configs/lazy.yaml --set twotower.lazy.encoder_layers=2 --run-dir artifacts/lazy/e2
python scripts/train_twotower.py --config configs/lazy.yaml --set twotower.lazy.mode=prompt --run-dir artifacts/lazy/prompt
python scripts/encode_twotower.py --run-dir artifacts/lazy/e0           # дальше — как у twotower
```

На Mac (mps) `embedding_bag` тегов не реализован — `PYTORCH_ENABLE_MPS_FALLBACK=1` (у v2 так же).
`online: true` для lazy не сделан (в Request нет времени каждого прослушивания).

## Ablation

| что | `--set` |
|---|---|
| слои энкодера | `twotower.lazy.encoder_layers=0 / 2 / 4` |
| реже cross-attention | `twotower.lazy.cross_every=2 / 4` |
| decoder-only бейзлайн | `twotower.lazy.mode=prompt` |
| без названия трека | `twotower.item.text.enabled=false` |
| без тегов | `twotower.tags.enabled=false` |
| последние треки вместо затухания | `twotower.input.history_order=recent` |
| длина истории | `twotower.input.history_len=64` (или `encode_twotower.py --history-len 64` без переобучения) |

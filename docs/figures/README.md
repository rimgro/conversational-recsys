# Схемы архитектур

Подробные описания к схемам: [docs/arch/](../arch/) — `twotower_v2.md` (и v1), `twotower_lazy.md`, `embeddings.md`.

Схемы описаны на [D2](https://d2lang.com) и собираются в SVG / PNG.

| файл | что |
|---|---|
| `twotower_v1.d2` | two-tower v1 (первая модель): только коды, раздельные таблицы |
| `twotower_v2.d2` | two-tower v2 (`recsys/twotower/model.py`): история в промпте Qwen, общие таблицы, короткий путь |
| `twotower_lazy.d2` | lazy (ветка `dev/lazy`, `recsys/twotower/lazy.py`): память истории + cross-attention в Qwen |
| `embeddings.d2` | откуда эмбеддинги трека: MuQ → RQ-VAE → коды, теги, название |
| `style.d2` | цвета и стили (классы `input`, `frozen`, `lora`, `trainable`, `emb`, …) |
| `legend.d2` | легенда, подключается в схемы как `legend: @legend` |

Сборка:

```bash
brew install d2
make -C docs/figures        # SVG
make -C docs/figures png    # + PNG
make -C docs/figures watch F=twotower_lazy   # предпросмотр с пересборкой при сохранении
```

Как править: блок — `имя: "подпись" {class: trainable}`, стрелка — `a -> b: "подпись"`, вложенность — фигурные
скобки (`towers.user.body.dec.xattn`). Цвет класса меняется в `style.d2` сразу во всех схемах.
Раскладка — grid (`grid-columns`): ряды сверху вниз — скор, башни (пользователь слева), эмбеддинги, входы;
порядок блоков в ряду = порядок в файле.

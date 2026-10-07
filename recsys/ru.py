"""Русский словарь для разбора запросов без LLM: основа слова -> английские теги Last.fm и атрибуты.

Запросы датасета на русском, теги каталога на английском. Совпадение по префиксу токена ловит
падежи и роды ('спокойный', 'спокойную', 'спокойно' -> 'спокойн'). Сначала ищутся фразы.

Значение записи: dict с ключами
  tags      английские теги
  country   код страны (artist_country в каталоге)
  lang      код языка текста (lang в каталоге)
  positive  True — фраза сама задаёт полярность ('без слов' = instrumental, а не исключение)
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

Entry = Dict[str, object]


def _t(*tags: str, **kw) -> Entry:
    return {"tags": list(tags), **kw}


# фразы: последовательность основ -> запись (длинные проверяются раньше)
PHRASES: List[Tuple[Tuple[str, ...], Entry]] = [
    (("без", "слов"), _t("instrumental", positive=True)),
    (("без", "вокал"), _t("instrumental", positive=True)),
    (("без", "текст"), _t("instrumental", positive=True)),
    (("женск", "вокал"), _t("female vocalists")),
    (("женск", "голос"), _t("female vocalists")),
    (("мужск", "вокал"), _t("male vocalists")),
    (("мужск", "голос"), _t("male vocalists")),
    (("хип", "хоп"), _t("hip hop")),
    (("трип", "хоп"), _t("trip hop")),
    (("нью", "эйдж"), _t("new age")),
    (("нью", "вейв"), _t("new wave")),
    (("нью", "йорк"), _t("new york", country="US")),
    (("синти", "поп"), _t("synthpop")),
    (("брит", "поп"), _t("britpop", country="GB")),
    (("инди", "рок"), _t("indie rock")),
    (("инди", "поп"), _t("indie pop")),
    (("поп", "рок"), _t("pop rock")),
    (("панк", "рок"), _t("punk rock")),
    (("пост", "панк"), _t("post punk")),
    (("пост", "рок"), _t("post rock")),
    (("хард", "рок"), _t("hard rock")),
    (("хеви", "метал"), _t("heavy metal")),
    (("классическ", "рок"), _t("classic rock")),
    (("прог", "рок"), _t("progressive rock")),
    (("прогрессив", "рок"), _t("progressive rock")),
    (("альтернатив", "рок"), _t("alternative rock")),
    (("гаражн", "рок"), _t("garage rock")),
    (("психодел", "рок"), _t("psychedelic rock")),
    (("софт", "рок"), _t("soft rock")),
    (("драм", "н", "бейс"), _t("drum n bass")),
    (("драм", "энд", "бейс"), _t("drum n bass")),
    (("ритм", "н", "блюз"), _t("rnb")),
    (("смус", "джаз"), _t("smooth jazz")),
    (("лоу", "фай"), _t("lo fi")),
    (("чил", "хоп"), _t("chill hop")),
    (("электронн", "музык"), _t("electronic")),
    (("классическ", "музык"), _t("classical")),
    (("для", "сна"), _t("sleep")),
    (("на", "русском"), {"tags": [], "lang": "ru"}),
    (("на", "английском"), {"tags": [], "lang": "en"}),
    (("на", "испанском"), {"tags": [], "lang": "es"}),
    (("на", "французском"), {"tags": [], "lang": "fr"}),
    (("на", "немецком"), {"tags": [], "lang": "de"}),
    (("на", "итальянском"), {"tags": [], "lang": "it"}),
    (("на", "португальском"), {"tags": [], "lang": "pt"}),
    (("на", "японском"), {"tags": [], "lang": "ja"}),
    (("на", "корейском"), {"tags": [], "lang": "ko"}),
]

# одиночные основы (префикс токена)
STEMS: Dict[str, Entry] = {
    # жанры
    "рок": _t("rock"), "поп": _t("pop"), "попс": _t("pop"), "джаз": _t("jazz"), "блюз": _t("blues"),
    "фолк": _t("folk"), "кантри": _t("country"), "рэп": _t("rap", "hip hop"), "реп": _t("rap", "hip hop"),
    "хипхоп": _t("hip hop"), "электрон": _t("electronic"), "техно": _t("techno"), "хаус": _t("house"),
    "транс": _t("trance"), "диско": _t("disco"), "фанк": _t("funk"), "соул": _t("soul"), "панк": _t("punk"),
    "метал": _t("metal"), "инди": _t("indie"), "альтернатив": _t("alternative"), "эмбиент": _t("ambient"),
    "амбиент": _t("ambient"), "регги": _t("reggae"), "гранж": _t("grunge"), "шугейз": _t("shoegaze"),
    "лаундж": _t("lounge"), "лундж": _t("lounge"), "кельт": _t("celtic"), "латин": _t("latin"),
    "индастриал": _t("industrial"), "дабстеп": _t("dubstep"), "саундтрек": _t("soundtrack"),
    "госпел": _t("gospel"), "шансон": _t("chanson"), "свинг": _t("swing"), "босса": _t("bossa nova"),
    "классик": _t("classical"), "классическ": _t("classic"), "синтипоп": _t("synthpop"),
    "грайм": _t("grime"), "трэп": _t("trap"), "драмэндбейс": _t("drum n bass"), "чилхоп": _t("chill hop"),
    "психодел": _t("psychedelic"), "прогрессив": _t("progressive"), "хардкор": _t("hardcore"),
    "ска": _t("ska"), "эмо": _t("emo"), "бит": _t("beats"), "андеграунд": _t("underground"),
    "андеграундн": _t("underground"), "эксперимент": _t("experimental"), "авангард": _t("avant garde"),
    # инструменты и вокал
    "инструментал": _t("instrumental"), "акустик": _t("acoustic"), "акустическ": _t("acoustic"),
    "оркестр": _t("orchestral"), "фортепиан": _t("piano"), "пианин": _t("piano"), "гитар": _t("guitar"),
    "саксофон": _t("saxophone"), "синтезатор": _t("synth"), "синтезаторн": _t("synth"),
    # настроение и энергия
    "спокойн": _t("calm", "chill"), "расслаб": _t("chill", "relaxing"), "умиротвор": _t("calm"),
    "мягк": _t("mellow"), "медленн": _t("slow"), "тих": _t("quiet"), "меланхол": _t("melancholic"),
    "грустн": _t("sad"), "печальн": _t("sad"), "тоскл": _t("sad"), "депресс": _t("depressing", "sad"),
    "весел": _t("happy"), "радост": _t("happy"), "позитив": _t("happy"), "жизнерадост": _t("happy"),
    "бодр": _t("energetic"), "энергичн": _t("energetic"), "драйв": _t("energetic"), "заряжа": _t("energetic"),
    "быстр": _t("fast"), "агрессив": _t("aggressive"), "яростн": _t("aggressive"), "зл": _t("angry"),
    "тяжел": _t("heavy"), "громк": _t("loud"), "танцевальн": _t("dance"), "танц": _t("dance"),
    "романтич": _t("romantic"), "нежн": _t("romantic"), "мечтательн": _t("dreamy"), "мрачн": _t("dark"),
    "темн": _t("dark"), "атмосферн": _t("atmospheric"), "эпичн": _t("epic"), "ностальг": _t("nostalgic"),
    "сексуальн": _t("sexy"), "чувствен": _t("sexy"), "цепля": _t("catchy"), "прилипчив": _t("catchy"),
    "плавн": _t("smooth"), "летн": _t("summer"), "зимн": _t("winter"), "ночн": _t("night"),
    "осенн": _t("autumn"), "уличн": _t("urban"), "красив": _t("beautiful"), "гипнотич": _t("hypnotic"),
    # ситуации
    "пробежк": _t("running", "energetic"), "бег": _t("running", "energetic"),
    "тренировк": _t("workout", "energetic"), "спортзал": _t("workout", "energetic"),
    "качалк": _t("workout", "energetic"), "спорт": _t("workout", "energetic"),
    "вечеринк": _t("party"), "тусовк": _t("party"), "учеб": _t("study", "calm"), "учёб": _t("study", "calm"),
    "концентрац": _t("focus", "calm"), "сосредоточ": _t("focus", "calm"), "засып": _t("sleep", "calm"),
    "уснуть": _t("sleep", "calm"), "медитац": _t("meditation", "calm"), "дорог": _t("driving", "road trip"),
    "поездк": _t("road trip", "driving"), "путешеств": _t("travel", "road trip"), "вождени": _t("driving"),
    "прогулк": _t("walking"), "свидани": _t("romantic", "love"),
    # эпохи (словами; цифры разбираются отдельно)
    "пятидесят": _t("50s"), "шестидесят": _t("60s"), "семидесят": _t("70s"), "восьмидесят": _t("80s"),
    "девяност": _t("90s"), "нулев": _t("00s"), "двухтысячн": _t("00s"),
    # страны
    "американ": _t("american", country="US"), "британ": _t("british", country="GB"),
    "английск": _t("british", country="GB"), "немецк": _t("german", country="DE"),
    "французск": _t("french", country="FR"), "шведск": _t("swedish", country="SE"),
    "японск": _t("japanese", country="JP"), "русск": _t("russian", country="RU"),
    "испанск": _t("spanish", country="ES"), "итальянск": _t("italian", country="IT"),
    "канадск": _t("canadian", country="CA"), "ирландск": _t("irish", country="IE"),
    "австралийск": _t("australian", country="AU"), "бразильск": _t("brazilian", country="BR"),
    "корейск": _t("korean", country="KR"), "финск": _t("finnish", country="FI"),
    "норвежск": _t("norwegian", country="NO"), "исландск": _t("icelandic", country="IS"),
    "польск": _t("polish", country="PL"), "голландск": _t("dutch", country="NL"),
    "мексиканск": _t("mexican", country="MX"), "ямайск": _t("jamaican", country="JM"),
    "калифорн": _t("california", country="US"), "лондон": _t("london", country="GB"),
    "манчестер": _t("manchester", country="GB"), "ливерпул": _t("british", country="GB"),
    "брит": _t("british", country="GB"), "бритпоп": _t("britpop", country="GB"),
    "русскоязычн": {"tags": [], "lang": "ru"}, "англоязычн": {"tags": [], "lang": "en"},
}
_STEMS_BY_LEN = sorted(STEMS, key=len, reverse=True)

# латинские слова страны: 'folk blues 90s usa'
COUNTRIES_EN: Dict[str, str] = {
    "usa": "US", "us": "US", "american": "US", "america": "US", "uk": "GB", "british": "GB", "english": "GB",
    "england": "GB", "german": "DE", "germany": "DE", "french": "FR", "france": "FR", "swedish": "SE",
    "japanese": "JP", "russian": "RU", "spanish": "ES", "italian": "IT", "canadian": "CA", "irish": "IE",
    "australian": "AU", "brazilian": "BR", "korean": "KR",
}

NEGATIONS_RU = frozenset("без не кроме никаких нет ни исключая".split())
# короткие основы совпадают только с целым токеном (иначе 'зл' ловит 'зло', 'ска' — 'скачать', ...)
_EXACT_ONLY = {"брит": ("брит",), "зл": ("злой", "злая", "злое", "злые", "злую", "злых"), "ска": ("ска",), "эмо": ("эмо",),
               "бег": ("бег", "бега", "бегу", "бегом"), "бит": ("бит", "биты", "битов", "битами"),
               "тих": ("тихий", "тихая", "тихое", "тихие", "тихую", "тихих"), "танц": ("танцев", "танцы", "танцами"),
               "поп": ("поп",), "реп": ("реп",), "транс": ("транс",), "спорт": ("спорт", "спорта")}

_DECADE_RE = re.compile(r"^(19|20)?([0-9])0(s|х|е|x)?$")


def lookup_stem(token: str) -> Optional[Entry]:
    for stem in _STEMS_BY_LEN:
        if token.startswith(stem):
            exact = _EXACT_ONLY.get(stem)
            if exact is not None and token not in exact:
                continue
            return STEMS[stem]
    # сравнительная степень с «по-» в диалоге: 'поэнергичнее', 'повеселее', 'побыстрее' -> без приставки
    if token.startswith("по") and token.endswith(("ее", "ей")) and len(token) > 6:
        return lookup_stem(token[2:])
    return None


def match_phrase(tokens: List[str], i: int) -> Optional[Tuple[int, Entry]]:
    """Самая длинная фраза из PHRASES, начинающаяся с tokens[i] -> (длина, запись)."""
    best = None
    for stems, entry in PHRASES:
        n = len(stems)
        if i + n <= len(tokens) and all(tokens[i + k].startswith(stems[k]) for k in range(n)):
            if best is None or n > best[0]:
                best = (n, entry)
    return best


def decade_tag(year: int) -> str:
    """1983 -> '80s', 2004 -> '00s', 2012 -> '10s' (как в тегах Last.fm)."""
    return f"{(int(year) // 10 % 10)}0s"


def match_era(tokens: List[str], i: int) -> Optional[Tuple[int, str, Optional[int]]]:
    """'80-х' / '80s' / '1980-х' / '2003' -> (число токенов, тег десятилетия, год или None)."""
    tok = tokens[i]
    nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
    if re.fullmatch(r"(19[5-9]|20[0-2])[0-9]", tok):  # четыре цифры
        year = int(tok)
        if nxt in ("х", "е", "x", "s") and year % 10 == 0:
            return 2, decade_tag(year), None
        return 1, decade_tag(year), year
    m = _DECADE_RE.match(tok)
    if m and (m.group(3) or nxt in ("х", "е", "x")):
        decade = int(m.group(2))
        if m.group(1) is None and decade == 0 and not m.group(3):
            return None
        consumed = 1 if m.group(3) else 2
        return consumed, f"{decade}0s", None
    return None


# ---------------------------------------------------------------- транслит (поиск по названию: «кате буш» -> kate bush)

_RU2LAT = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh", "з": "z", "и": "i", "й": "y",
           "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
           "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e",
           "ю": "yu", "я": "ya"}

# слова запроса, которые точно не часть названия
STOPWORDS_RU = frozenset("""
песня песню песни песен трек трека треки музыка музыку музыки группа группы исполнитель альбом
где что это как про со с в на и или по для из от до у к о об мне хочу найди найти включи поставь
поют поет поёт словами слова слов строчка строчкой фраза фразу помню только там тот та который которая
""".split())


def translit(token: str) -> str:
    return "".join(_RU2LAT.get(ch, ch) for ch in token)

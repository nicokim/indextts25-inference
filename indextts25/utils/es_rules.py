"""Spanish rules around NeMo's text normalization, tuned for Argentine (Rioplatense) chat text.

``normalize(text, nemo)`` is ``post(nemo(pre(text)))``. The pre-rules handle what NeMo's
Spanish grammar reads wrong:

- Plain integers are spelled here, with agreement. NeMo always apocopates ("Capítulo un",
  "a la un", "ciento un personas") and misreads any number containing 800 before a word
  ("2.800 perros" -> "dos ochocientos perros", "1.800 km hasta" -> "uno ochocientos km"), so
  integers with a unit ("5 km", "300kg") are spelled here too.
- Money: ``$`` is pesos (the Latin American convention), before or after the amount;
  ``US$``/``U$S``/``USD`` are dollars, also as "USD $500" or "$500 USD"; cents go after
  "con": "$1.250,50" -> "mil doscientos cincuenta pesos con cincuenta". "$1,000.50" (US
  style) works too.
- Ordinals ("1era", "2dos", "9no", "1º puesto" -> "primer puesto"), times written with a period
  ("21.30 hs", "nos vemos 9.30"), heights ("mide 1,70" -> "uno setenta"), scores and ranges
  ("3-1" -> "tres a uno", "1-3 horas" -> "una a tres horas"; NeMo reads a date), "vs", "2x1",
  "24/7", ratings ("8/10"), decimals and versions written with a period ("13.8", "2.1.3").
- What NeMo leaves raw: GB/MB/TB, "k"/"M" multipliers, "hs", degrees, decades ("los 90s"),
  digits glued to letters ("MatiGamer99", "F1", "3D", "4K"), handles with "_" ("Mati_99"),
  mails and URLs (NeMo spells them letter by letter).
- Roman numerals are read only in clear contexts: after "siglo", "capítulo"..., after a regnal
  name ("Juan Pablo II" -> "segundo", "Isabel II" -> "segunda", "Luis XIV" -> "catorce":
  ordinals up to ten), in "del I al XII", and after a title ("Final Fantasy XIV" -> "catorce",
  two letters or more). Any other Roman-looking token ("MI", "CV", "GTA V", "vitamina D", the
  initial of "Juan D. Perón"), "S"/"N" (NeMo: "sur", "norte") and "da." ("doña") get an
  invisible word joiner, which no NeMo grammar matches and which is removed afterwards.

The post-rules remove the space NeMo leaves before punctuation and fix the number words it
still apocopates inside times and decimals ("trece y un a casa" -> "trece y uno a casa").
"""

import re
from collections.abc import Callable

_WJ = "\u2060"  # WORD JOINER
_L = "A-Za-zÁÉÍÓÚÜÑáéíóúüñ"  # letters; \w would also take digits and "º"

# ---------------------------------------------------------------------------------------------
# Spelling
# ---------------------------------------------------------------------------------------------
_ONES = (
    "cero uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce quince dieciséis "
    "diecisiete dieciocho diecinueve veinte veintiuno veintidós veintitrés veinticuatro veinticinco "
    "veintiséis veintisiete veintiocho veintinueve"
).split()
_TENS = "treinta cuarenta cincuenta sesenta setenta ochenta noventa".split()
_HUNDREDS = (
    "ciento doscientos trescientos cuatrocientos quinientos seiscientos setecientos ochocientos novecientos".split()
)
_ORDINALS = "primero segundo tercero cuarto quinto sexto séptimo octavo noveno décimo".split()
_SCALES = ("millón", "millones", "billón", "billones")
_APOCOPE = {"uno": "un", "veintiuno": "veintiún", "una": "un", "veintiuna": "veintiún"}
_TO_FEM = {"uno": "una", "veintiuno": "veintiuna", **{h: h[:-2] + "as" for h in _HUNDREDS[1:]}}


def _apocope(words: list[str]) -> list[str]:
    return words[:-1] + [_APOCOPE.get(words[-1], words[-1])]


def _words(n: int) -> list[str]:
    out: list[str] = []
    for scale, one, many in ((10**12, "billón", "billones"), (10**6, "millón", "millones")):
        q, n = divmod(n, scale)
        if q:
            out += ["un", one] if q == 1 else _apocope(_words(q)) + [many]
    q, n = divmod(n, 1000)
    if q:
        out += ["mil"] if q == 1 else _apocope(_words(q)) + ["mil"]
    h, r = divmod(n, 100)
    if h:
        out.append("cien" if n == 100 else _HUNDREDS[h - 1])
    if r >= 30:
        out += [_TENS[r // 10 - 3]] + (["y", _ONES[r % 10]] if r % 10 else [])
    elif r:
        out.append(_ONES[r])
    return out or ["cero"]


def _spell(n: int, gender: str = "") -> str:
    """Cardinal. ``gender`` "" is the full form ("uno", "veintiuno"); "m"/"f" the form before a
    masculine/feminine noun ("un", "veintiún"; "una", "doscientas"). Millions stay masculine."""
    words = _words(n)
    if gender == "m":
        words = _apocope(words)
    elif gender == "f":
        cut = max((i + 1 for i, w in enumerate(words) if w in _SCALES), default=0)
        words = words[:cut] + [_TO_FEM.get(w, w) for w in words[cut:]]
    return " ".join(words)


def _digits(s: str) -> str:
    return " ".join(_ONES[int(c)] for c in s)


def _fraction(s: str) -> str:
    """Digits after a decimal point: leading zeros one by one, then up to two digits as a number."""
    rest = s.lstrip("0")
    words = [_digits(s[: len(s) - len(rest)]), _spell(int(rest)) if 0 < len(rest) <= 2 else _digits(rest)]
    return " ".join(w for w in words if w)


def _height(m: re.Match) -> str:
    """'1,70' -> 'uno setenta', '1.05' -> 'uno cero cinco'."""
    a, b = m.group("m"), m.group("cm")
    return (m.groupdict().get("pre") or "") + f"{_spell(int(a))} {_digits(b) if b[0] == '0' else _spell(int(b))}"


# ---------------------------------------------------------------------------------------------
# What surrounds a number
# ---------------------------------------------------------------------------------------------
# A number before these does not count them, so it keeps its full form ("tres a uno", "21 en").
_CLOSED = frozenset(
    """
a al ante bajo con contra de del desde durante en entre hacia hasta para por según sin sobre tras versus vía
y e o u ni pero sino que porque pues aunque si como cuando donde mientras entonces
el la los las lo un una unos unas este esta estos estas ese esa esos esas mi mis tu tus su sus cada
yo vos tú él ella ellos ellas nosotros me te se le les nos usted ustedes todos todas ambos ambas otros otras
no sí ya más menos muy también tampoco nomás solamente apenas casi ahora hoy ayer siempre nunca jamás
acá allá ahí aquí allí así todavía aún después antes luego recién igual nada algo todo
es son era eran fue fueron será serán sería sea sean está están estaba estaban estuvo hay había hubo habrá
tiene tienen tenía tenían tuvo tengo tenés tenemos va van iba iban vamos voy viene vienen vino sale salen
queda quedan pasa pasan dice dicen hace hacen puede pueden sigue siguen vale valen cuesta cuestan
soy sos eres somos estoy estás estamos dijo hizo puso trajo quiso supo pudo anduvo gracias
entero entera enteros enteras completo completa completos completas
global mundial general total nacional oficial especial normal ideal internacional local
""".split()
)
# Verb endings no counted noun has ("guardan", "jugando", "ganó"), minus the nouns that do.
_VERB = re.compile(rf"(?:ando|iendo|yendo|aron|ieron|aban|ían|[{_L}]{{3}}(?:an|en)|[{_L}]ó)$")
_NOUN_EN = frozenset("imagen examen volumen origen orden crimen joven virgen resumen margen abdomen régimen".split())
# A number right after these is a label, read in full: "capítulo uno", "área cincuenta y uno".
_LABELS = frozenset(
    """
área area villa número numero nro núm capítulo capitulo página pagina pág person persona fila ruta
habitación piso puerta mesa línea linea canal calle avenida distrito sector nivel temporada episodio parte
tomo volumen round ronda jugador player top puesto versión version modelo año siglo kilómetro km minuto
lote zona sala aula asiento vagón andén vuelo colectivo curso categoría artículo ley paso opción pregunta
ejercicio juego equipo grupo team fase etapa día semana camiseta edición
""".split()
)
_LABEL_ABBR = re.compile(r"(?i)\b(?:pág|pag|cap|art|vol|ep|temp|nro|núm|num)\.\s*$")  # "pág. 21"
_LABEL_NEXT = re.compile(rf"\s+[{_L}]+\.?\s*\d")  # "capítulo 1 parte 2": the next word is a label too
_FEM_NOUNS = frozenset(
    """
vez veces noche noches mujer mujeres foto fotos moto motos mano manos calle calles parte partes clase clases
red redes imagen imágenes ley leyes flor flores llave llaves torre torres tarde tardes muerte muertes
frase frases serie series radio radios piel pieles luz luces voz voces razón razones
crisis tesis dosis hipótesis síntesis kill kills skin skins build builds run runs
vista vistas revista revistas pista pistas lista listas entrevista entrevistas conquista conquistas
""".split()
)
_MASC_A = frozenset(
    """
día días mapa mapas problema problemas tema temas sistema sistemas programa programas idioma idiomas clima
planeta planetas poema poemas drama dramas fantasma fantasmas pijama pijamas cometa cometas dilema dilemas
esquema esquemas síntoma síntomas trauma traumas enigma enigmas diagrama diagramas aroma aromas
giga gigas mega megas tera teras dálmata dálmatas idiota idiotas hincha hinchas policía policías
pirata piratas atleta atletas poeta poetas colega colegas astronauta astronautas psicópata psicópatas
pana panas gorila gorilas panda pandas koala koalas puma pumas avión aviones camión camiones
""".split()
)
# Feminine nouns that take "un" in the singular: "un arma", "un águila"
_TONIC_A = frozenset("arma alma águila agua área aula hacha hambre ala ancla ave hada habla álgebra arpa alba".split())
# Singular nouns ending in -a/-e that a 1 after a label or a name still counts ("Kimox 1 luca")
_COUNT_SG = frozenset(
    "partida luca birra cerveza pizza empanada idea palabra pregunta cosa persona hora semana moneda vida "
    "carta ronda mina mensaje like chiste golpe viaje".split()
)
# Skipped to find the noun: "1 gran idea", "1 sola vez"; "Tengo 1 solo." has none
_ADJ_PRE = frozenset("gran mejor peor solo sola solos solas".split())
_FEM_END = re.compile(r"(?:a|as|ión|iones|dad|dades|tud|tudes|umbre|umbres)$")
_MONTH_NAMES = "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split()
_MONTHS = "|".join(_MONTH_NAMES + ["setiembre"])
_UNIT = rf"\s?(?:km²|m²|m³|°|%|kph|mph|kg|km|cm|mm|ml|ms|min|am|pm|AM|PM|a\.m\.|p\.m\.|[ghlms])(?![{_L}])"
# Left to NeMo: "1 de mayo" (dates), "5 %", "3am" (times).
_NEMO_AFTER = re.compile(rf"\s+de\s+(?i:{_MONTHS})\b|{_UNIT}")
_NEXT = re.compile(rf"\s+([{_L}]+)")
_COORD = re.compile(rf"\s+[oueya]\s+\d[\d.]*\s+([{_L}]+)")  # "1 o 2 veces", "de 1 a 3 horas"
_PREV = re.compile(rf"([{_L}]+)\s+$")
_UNIT_AT = re.compile(_UNIT)
_BACK = 60  # how far back the rules look for the previous word; keeps them linear in long texts


def _prev_word(text: str, pos: int) -> str:
    m = _PREV.search(text, max(0, pos - _BACK), pos)
    return m.group(1).lower() if m else ""


def _is_fem(word: str) -> bool:
    w = word.lower()
    return w in _FEM_NOUNS or (w not in _MASC_A and not w.endswith(("ista", "istas")) and bool(_FEM_END.search(w)))


def _counted(text: str, end: int) -> tuple[str, str]:
    """(noun, adjective) for a number ending at ``text[end]``: "1 perro" -> ("perro", ""), "1 sola
    vez" -> ("vez", "sola"), "1 solo." -> ("", "solo"); no noun before a verb, a function word or
    punctuation."""
    nxt = _NEXT.match(text, end)
    word = nxt.group(1).lower() if nxt else ""
    adj = ""
    if word in _ADJ_PRE:
        adj, nxt = word, _NEXT.match(text, nxt.end())
        word = nxt.group(1).lower() if nxt else ""
    if not word or word in _CLOSED or (word not in _NOUN_EN and _VERB.search(word)):
        return "", adj
    return word, adj


def _counts_after_label(n: int, noun: str, text: str, end: int) -> bool:
    """After a label or a name, a number counts only a noun that clearly is one: a plural
    ("ToxicMati 100 bits"), or for 1 a singular that is no verb form ("Kimox 1 bit", not
    "Capítulo 1 empieza")."""
    if not noun or (noun in _LABELS and _LABEL_NEXT.match(text, end)):
        return False
    if n != 1:
        return noun.endswith("s")
    return noun[-1] not in "ae" or noun in _FEM_NOUNS or noun in _COUNT_SG


def _say(n: int, text: str, start: int, end: int) -> str:
    """Spell the number ``n`` found at ``text[start:end]`` the way its neighbors ask for."""
    nxt = _NEXT.match(text, end)
    word = nxt.group(1) if nxt else ""
    lw = word.lower()
    noun, adj = _counted(text, end)
    prev = _PREV.search(text, max(0, start - _BACK), start)
    pw = prev.group(1) if prev else ""
    head = text[max(0, prev.start() - _BACK) : prev.start()].rstrip() if prev else ""
    # "Capítulo 1", "el Area 51 guardan": a label, not a count (a capital word is a name unless
    # it starts the sentence and is not all caps like "FIFA 21", or the whole text is shouted)
    name = (
        pw[:1].isupper()
        and not (len(word) > 1 and word.isupper())
        and ((head and head[-1] not in ".!?¡¿…:;'\"«(") or (len(pw) > 1 and pw.isupper()))
    )
    if (
        pw.lower() in _LABELS or name or _LABEL_ABBR.search(text, max(0, start - _BACK), start)
    ) and not _counts_after_label(n, noun, text, end):
        return _spell(n)
    if lw == "mil":  # "21 mil" -> "veintiún mil"; "200 mil personas" agrees with the noun after it
        after = _counted(text, nxt.end())[0]
        return " ".join(_apocope(_spell(n, "f" if after and _is_fem(after) else "m").split()))
    if not noun:
        if adj:  # "una sola", "uno solo"
            return _spell(n, "f" if adj.endswith(("a", "as")) else "")
        coord = _COORD.match(text, end)
        if (coord and _is_fem(coord.group(1))) or pw.lower() in ("la", "las"):
            return _spell(n, "f")  # "una o dos veces", "a la una", "a las veintiuna"
        return _spell(n)
    words = _spell(n, "f" if _is_fem(noun) and noun not in _TONIC_A else "m")
    return words + " de" if words.endswith(_SCALES) else words  # "un millón de personas"


def _code(digits: str) -> str | None:
    """Digit strings read one digit at a time: leading zeros ("07"; "0800" is "cero ochocientos")
    and long unseparated runs that are not round (phone numbers, ids). None for plain numbers."""
    if len(digits) > 1 and digits[0] == "0":
        toll_free = re.match(r"0(?:800|810|600)", digits)
        return f"cero {_spell(int(digits[1:4]))} {_digits(digits[4:])}".rstrip() if toll_free else _digits(digits)
    return _digits(digits) if len(digits) > 6 and not digits.endswith("000") else None


def _phone(m: re.Match) -> str:
    """'11 4567-8901' -> 'once cuatro cinco seis siete ocho nueve cero uno'; '0800' -> 'cero ochocientos'."""
    groups = re.findall(r"\d+", m.group(0))
    words = [_spell(int(g)) if len(g) <= 2 and g[0] != "0" else _code(g) or _digits(g) for g in groups]
    return ("más " if m.group(0).startswith("+") else "") + " ".join(words)


def _integer(m: re.Match) -> str:
    s, text = m.group(0), m.string
    digits = s.replace(".", "")
    if _NEMO_AFTER.match(text, m.end()) or len(digits) > 18:
        return s
    # a long run is a code, unless it counts something: "1234567 vistas"
    if "." not in s and (code := _code(digits)) and not (digits[0] != "0" and _counted(text, m.end())[0][-1:] == "s"):
        return code
    return _say(int(digits), text, m.start(), m.end())


# ---------------------------------------------------------------------------------------------
# Rule callbacks
# ---------------------------------------------------------------------------------------------
_MULT = {"k": 10**3, "K": 10**3, "mil": 10**3, "M": 10**6, "millón": 10**6, "millones": 10**6}


def _scaled(whole: str, dec: str | None, mult: str) -> int:
    """'2', '5', 'K' -> 2500."""
    return int(whole) * _MULT[mult] + (int(dec) * _MULT[mult] // 10 ** len(dec) if dec else 0)


def _money(m: re.Match) -> str:
    raw = re.sub(r"[.,]", "", m["n"])
    if len(raw) > 18:
        return m.group(0)
    n, cents, mult = int(raw), m["c"], m["mult"]
    if mult:  # "$1,5 millones": the decimals belong to the multiplier
        n, cents = _scaled(raw, cents, mult), None
    cur = (m.groupdict().get("cur2") or m["cur"]).lower()
    if cur in ("$", "ar$", "ars", "pesos"):
        one, many = "peso", "pesos"
    elif cur in ("€", "euros"):
        one, many = "euro", "euros"
    else:
        one, many = "dólar", "dólares"
    c = int(cents.ljust(2, "0")) if cents else 0
    if n == 0 and c:
        return f"{_spell(c, 'm')} {'centavo' if c == 1 else 'centavos'}"
    words = _spell(n, "m")
    words += f"{' de' if words.endswith(_SCALES) else ''} {one if n == 1 else many}"
    return f"{words} con {_spell(c)}" if c else words


_KILO_COUNT = frozenset(
    "seguidores suscriptores subs views vistas visitas bits followers likes viewers espectadores personas".split()
)


def _kilo(m: re.Match) -> str:
    """'2.5K' -> '2.500', spelled later with its context; '4K' and '8K' are resolutions, unless
    they count followers ('4k subs')."""
    whole, dec, mult, de = m.groups()
    if mult in "kK" and not dec and whole in ("4", "8"):
        nxt = _NEXT.match(m.string, m.end())
        if not (nxt and nxt.group(1).lower() in _KILO_COUNT):
            return f"{whole} K{de or ''}"
    return f"{_scaled(whole, dec, mult):,}".replace(",", ".") + (de if mult == "M" and de else "")


def _data(m: re.Match) -> str:
    word = {"g": "giga", "m": "mega", "t": "tera", "k": "kilobyte"}[m.group(2)[0].lower()]
    return f"{m.group(1)} {word}{'' if m.group(1) == '1' else 's'}"


_ABBR_UNITS = {
    "hs": "hora",
    "hr": "hora",
    "hrs": "hora",
    "seg": "segundo",
    "segs": "segundo",
    "mins": "minuto",
    "mg": "miligramo",  # NeMo leaves "0,5 mg" as is
    "mts": "metro",
    "kms": "kilómetro",
}


def _abbr_unit(m: re.Match) -> str:
    one = re.search(r"(?<![\d.,])1$", m.string[max(0, m.start() - _BACK) : m.start()].rstrip())
    return f" {_ABBR_UNITS[m.group(1)]}{'' if one else 's'}"


# Measures. NeMo misreads 800 here too ("1.800 km hasta" -> "uno ochocientos km"); the words are NeMo's.
_MEASURES = {
    "km": "kilómetro",
    "m": "metro",
    "cm": "centímetro",
    "mm": "milímetro",
    "kg": "kilogramo",
    "g": "gramo",
    "ml": "mililitro",
    "l": "litro",
    "ms": "milisegundo",
    "min": "minuto",
    "s": "segundo",
    "h": "hora",
    "km²": "kilómetro cuadrado",
    "km2": "kilómetro cuadrado",
    "m²": "metro cuadrado",
    "m2": "metro cuadrado",
    "m³": "metro cúbico",
    "m3": "metro cúbico",
    "km/h": "kilómetro por hora",
    "kph": "kilómetro por hora",
    "mph": "milla por hora",
}


def _measure(m: re.Match) -> str:
    n, unit = m.group(1), m.group(2)
    one, _, rest = _MEASURES[unit.lower()].partition(" ")
    many = one + "s" + (" " + rest + ("s" if rest.endswith("o") else "") if rest else "")
    return f"{n} {one + (' ' + rest if rest else '') if n == '1' else many}"


_TIME_CUES = frozenset(
    "la las hoy mañana tipo desde hasta llego llegás llega llegamos arranca arranco arrancamos arrancás "
    "empieza empiezo empezamos vemos juntamos salgo salimos termina terminamos abre cierra".split()
)
_TIME_UNIT = re.compile(rf"\s?(?:hs|hrs?|h|horas|am|pm|AM|PM)(?![{_L}])|\s?[ap]\.\s?m\.")


def _dot_time(text: str, start: int, end: int, h: str, mm: str, cues: frozenset = _TIME_CUES) -> str | None:
    """'21.30 hs', 'nos vemos 9.30', 'a las 3.30': a time, written 'H:MM' for NeMo."""
    if len(mm) == 2 and int(h) <= 23 and int(mm) <= 59:
        cue = _prev_word(text, start) in cues and not _UNIT_AT.match(text, end)  # not "desde 3.50 kg"
        if cue or _TIME_UNIT.match(text, end):
            return f"{h}:{mm}"
    return None


def _decimal(a: str, b: str, text: str, end: int) -> str:
    """'13.8' -> 'trece punto ocho' (NeMo: thousands, or a time); '1.5x' -> 'uno punto cinco x'."""
    words = f"{_spell(int(a))} punto {_fraction(b)}"
    return words + " " if text[end : end + 1].isalpha() else words


def _period_decimal(m: re.Match) -> str:
    a, b, text = m[1], m[2], m.string
    if time := _dot_time(text, m.start(), m.end(), a, b):
        return time
    if _UNIT_AT.match(text, m.end()):
        return f"{a},{b}"  # before a unit, a Spanish comma: NeMo reads "quince coma cinco kilómetros"
    return _decimal(a, b, text, m.end())


_NOT_HEIGHT = re.compile(
    r"\s+(?:litros?|kilos?|kilogramos?|gramos?|veces|puntos?|goles?|pesos|dólares|euros|millones|horas?|años?)\b"
)


def _bare_height(m: re.Match) -> str:
    """'1.80' alone, with a period, is a height in chat ('galanes de 1.80', '1.55 perfectos')."""
    a, b, text = m["m"], m["cm"], m.string
    if time := _dot_time(text, m.start(), m.end(), a, b, frozenset(("la", "las"))):  # not "desde 1.50"
        return time
    if (
        text[m.end() : m.end() + 1].isalpha()
        or _NOT_HEIGHT.match(text, m.end())
        or _prev_word(text, m.start()) in ("versión", "version")
    ):
        return _decimal(a, b, text, m.end())  # "1.75 litros", "1.25x", "versión 1.50"
    return _height(m)


def _version(m: re.Match) -> str:
    """'2.1.3' -> 'dos punto uno punto tres'; '1.000.000' stays a number, '12.03.2026' a date."""
    parts = m.group(0).split(".")
    if all(len(p) == 3 for p in parts[1:]) or (
        len(parts) == 3 and int(parts[0]) <= 31 and 0 < int(parts[1]) <= 12 and len(parts[2]) in (2, 4)
    ):
        return m.group(0)
    return " punto ".join(_spell(int(p)) if p[0] != "0" or p == "0" else _digits(p) for p in parts)


def _range(m: re.Match) -> str:
    """'3-5 días' is a range of counts (spelled later, with agreement: 'una a tres horas'); '1-0
    arriba' a score and '1990-2000' a span, read in full."""
    a, b = m.group(1), m.group(2)
    if _counted(m.string, m.end())[0][-1:] == "s":
        return f"{a} a {b}"
    return f"{_spell(int(a))} a {_spell(int(b))}"


def _ordinal(m: re.Match) -> str:
    """'1era' -> 'primera', '2dos' -> 'segundos', '9no' -> 'noveno', '1º puesto' -> 'primer puesto'."""
    word, suffix = _ORDINALS[int(m.group(1)) - 1], m.group(2)
    plural = "s" if suffix.endswith("s") else ""
    if suffix.rstrip("s")[-1] in "aª":
        return word[:-1] + "a" + plural
    noun = _counted(m.string, m.end())[0]
    if word in ("primero", "tercero") and noun and not plural and noun[-1] != "s" and not _is_fem(noun):
        return word[:-1]
    return word + plural


_DEGREE_CUES = frozenset("hace hacía hacen hará hizo marca marcaba marcó marcan unos".split())
_DEGREE_AFTER = re.compile(r"\s+(?:bajo cero|de (?:calor|fiebre|temperatura|sensación)|a la sombra)\b")


def _degrees(m: re.Match) -> str:
    """'Hace 30°' -> 'Hace 30 grados' (NeMo: 'trigésimo'); '1º puesto' stays an ordinal."""
    sign, n, scale = m.groups()
    pw = _prev_word(m.string, m.start())
    if not (scale or pw in _DEGREE_CUES or (pw == "a" and int(n) > 10) or _DEGREE_AFTER.match(m.string, m.end())):
        return m.group(0)
    return f"{'menos ' if sign else ''}{n} grado{'' if n == '1' else 's'}"


_SUFFIXES = frozenset(
    "km kg cm mm ml ms m g h l min s mph kph am pm AM PM ro do to vo no mo er ra da ta va na ma".split()
)
_EXTENSIONS = "exe|com|net|org|ar|io|gg|tv|app|dev|ly|xyz|txt|pdf|jpg|png|gif|zip|rar|py|js"


def _web(tok: str) -> str:
    """Mails and URLs the way they are said: 'kohai.vtuber@gmail.com' -> 'kohai punto vtuber arroba
    gmail punto com', 'https://twitch.tv/kohai' -> 'twitch punto tv barra kohai'."""
    core, tail = re.fullmatch(r"(.*?)([.,;:!?)\]'\"»]*)", tok).groups()
    core = re.sub(r"^https?://|[?#].*", "", core)  # no query strings
    core = re.sub(r"^www\.", "doble ve doble ve doble ve punto ", core)
    core = re.sub(r"(?<=\w)\.(?=\w)", " punto ", core.replace("@", " arroba "))
    core = re.sub(r"(?<=\w)/(?=\w)", " barra ", core).rstrip("/")
    return core.replace("_", " guion bajo ").replace("-", " guion ") + tail


def _token(m: re.Match) -> str:
    """Spell digits glued to letters ("MatiGamer99", "3D", "C5N"), except before the suffixes
    NeMo reads (units, am/pm, ordinals); mails, URLs and files ("virus.exe") with "punto"."""
    tok = m.group(0)
    if re.search(rf"\w@\w|://|www\.|\w\.(?:{_EXTENSIONS})\b", tok):
        tok = _web(tok)

    def glued(g: re.Match) -> str:
        digits, letters = g.groups()
        after_letter = g.start() > 0 and re.match(rf"[{_L}]", tok[g.start() - 1])
        if not after_letter and (not letters or letters in _SUFFIXES):
            return g.group(0)
        words = _code(digits) or _spell(int(digits))
        return f"{' ' if after_letter else ''}{words}{' ' + letters if letters else ''}"

    return re.sub(rf"(\d+)([{_L}]*)", glued, tok)


def _underscores(m: re.Match) -> str:
    """'Mati_99' -> 'Mati 99' (the digits are spelled later); mails and URLs keep theirs."""
    tok = m.group(0)
    return tok if re.search(r"\w@\w|://|www\.", tok) else re.sub(r"(?<=[^\W_])_+(?=[^\W_])", " ", tok)


def _roman(s: str) -> int:
    """Value of a valid Roman numeral (either case), else 0."""
    s = s.upper()
    if not s or not re.fullmatch(r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})", s):
        return 0
    vals = [{"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}[c] for c in s]
    return sum(-v if v < w else v for v, w in zip(vals, vals[1:] + [0]))


_ROMAN = r"[IVXLCDM]+|[ivxlcdm]+"
_REGNAL = frozenset(
    """
Juan Pablo Luis Felipe Carlos Isabel Enrique Fernando Alfonso Benedicto Pío León Gregorio Pedro Jorge Eduardo
Guillermo Napoleón Ricardo Jaime Clemente Francisco Juana Federico Alejandro Constantino Nicolás Iván Sancho
Ramsés Inocencio Urbano Sixto Bonifacio Leopoldo Gustavo Carlota Victoria
""".split()
)
_FEM_REGNAL = frozenset("Isabel Juana Carlota Victoria".split())
_INITIAL = re.compile(r"\.\s+(?:[A-ZÁÉÍÓÚÑ]|de\b|del\b)")  # "Juan D. Perón", "Juan M. de Rosas"


def _roman_nouns(m: re.Match) -> str:
    """'siglo XXI' -> 'siglo veintiuno', 'siglos XV y XVI' -> 'siglos quince y dieciséis'."""
    if not _roman(m.group(2)):
        return m.group(0)
    more = re.sub(_ROMAN, lambda r: _spell(_roman(r.group(0))) if _roman(r.group(0)) else r.group(0), m.group(3))
    return m.group(1) + _spell(_roman(m.group(2))) + more


def _regnal(m: re.Match) -> str:
    name, num = m.group(1), m.group(2)
    v = _roman(num)
    if not v or name not in _REGNAL or (len(num) == 1 and _INITIAL.match(m.string, m.end())):
        return m.group(0)  # an initial is shielded below
    if v > 10:
        return f"{name} {_spell(v)}"
    word = _ORDINALS[v - 1]
    return f"{name} {word[:-1] + 'a' if name in _FEM_REGNAL else word}"


def _span(m: re.Match) -> str:
    a, b = _roman(m.group(2)), _roman(m.group(4))
    return f"{m.group(1)} {_spell(a)} {m.group(3)} {_spell(b)}" if a and b else m.group(0)


def _sequel(m: re.Match) -> str:
    """'Final Fantasy XIV' -> 'catorce'; not 'XXX', 'XL' or shouted text ('POR FIN VI EL CAPÍTULO')."""
    prev, num = m.group(1), m.group(2)
    v = _roman(num)
    after = _NEXT.match(m.string, m.end())
    shouted = len(prev) > 1 and prev.isupper() and after and len(after.group(1)) > 1 and after.group(1).isupper()
    if not v or prev.lower() in _CLOSED or num == "XXX" or ("L" in num and len(num) < 3) or shouted:
        return m.group(0)
    return f"{prev} {_spell(v)}"


_DON = re.compile(r"\.\s+(?!Do[nñ])[A-ZÁÉÍÓÚÑ][a-záéíóúñ]")


def _shield(m: re.Match) -> str:
    """Append the word joiner, except to the 'D.' of 'D. Juan Pérez' or 'a D. Juan', which NeMo
    reads 'Don' (not 'vitamina D. Juan dijo', nor the initial of 'Juan D. Perón')."""
    tok = m.group(1)
    if tok == "D" and _DON.match(m.string, m.end()):
        prev = _PREV.search(m.string, max(0, m.start() - _BACK), m.start())
        if not prev or prev.group(1) in _CLOSED:
            return tok
    return tok + _WJ


def _abbr(name: str) -> Callable[[re.Match], str]:
    # NeMo's own "EE. UU." -> "Estados Unidos" swallows the final period too; keep it at the end
    return lambda m: name + ("." if m.group(0).endswith(".") and m.end() == len(m.string) else "")


_RANGE_END = r"(0|[1-9]\d{0,2}|(?:1[5-9]|20)\d\d|[1-9]\d00)"  # up to 999, a year, or round hundreds
_ROMAN_NOUNS = "siglos?|capítulos?|tomos?|partes?|volumen|temporadas?|actos?|fases?|episodios?|libros?|niveles?"
_DOLLARS = r"US\$|U\$[SDsd]|USD|usd|u\$[sd]"
_AMOUNT = (
    r"(?P<n>\d{1,3}(?:\.\d{3})+|\d{1,3}(?:,\d{3})+|\d+)(?:[.,](?P<c>\d{1,2}))?(?!\d)(?:\.-)?"
    r"(?:\s?(?P<mult>mil(?:lones|lón)?|[kKM])\b)?"
)
_PRE: list[tuple[re.Pattern, str | Callable[[re.Match], str]]] = [
    (re.compile(p), r)
    for p, r in [
        (r"\d{19,}", lambda m: _digits(m.group(0))),  # an id, not a number (and int() has a limit)
        (r"\bRR\.?\s?HH\b\.?", _abbr("recursos humanos")),
        (r"\bEE\.?\s?UU\b\.?", _abbr("Estados Unidos")),
        (r"\betc\.", _abbr("etcétera")),
        (r"\baprox\.", _abbr("aproximadamente")),
        (r"\bDe?pto\.", _abbr("Departamento")),  # "Depto. 4B"; a bare "depto" is said as is
        (r"\bde?pto\.", _abbr("departamento")),
        (r"\b[Vv]ol\.\s?(?=[IVXLCDM]+\b|\d)", "volumen "),
        (r"\bc/u\b", "cada uno"),
        (r"\bc/\s?(?=[a-záéíóúñ])", "con "),
        (r"(?<!\S)\S*_\S*", _underscores),  # (?<!\S): each token is tried once
        (r"(?<![\w@])@(?=\w)", "arroba "),  # NeMo spells a handle letter by letter, and its comma as "coma"
        (r"(?<![\w#&])#(?=[A-Za-zÁÉÍÓÚÑáéíóúñ])", ""),  # hashtags
        (r"\b[Nn](?:ro\.?|\.?\s?[º°]|\.)\s?(?=\d)|\bnúm\.\s?(?=\d)|#\s?(?=\d)", "número "),
        (r"(?<!\d)(?<!\d )\b[Kk]m\.?\s?(?=\d)", "kilómetro "),  # "el km 21"
        (
            rf"(?<![\w$])(?P<cur>(?:{_DOLLARS})\s?\$?|AR\$|ARS|\$|€)\s?{_AMOUNT}"
            rf"(?:\s?(?P<cur2>{_DOLLARS}|dólares|dolares|pesos|euros)(?!\w))?(?![/:-]\d)",
            _money,
        ),
        (rf"(?<![\w.,/:$€-]){_AMOUNT}\s?(?P<cur>{_DOLLARS}|AR\$|ARS|\$|€)(?!\w)", _money),
        (r"(?<=\w)\s?/\s?(mes|día|dia|hora|año|semana|noche|persona|kilo|unidad)\b", r" por \1"),  # "$1.500/mes"
        (r"\b24/7\b", "veinticuatro siete"),
        (r"\b50/50\b", "cincuenta y cincuenta"),
        (r"\b(una?\s+)(\d{1,2})/10\b", r"\1\2 de 10"),  # "un 15/10" is a rating, not a date
        # "el 31/12": NeMo reads a day and month without a year as a fraction ("treinta y un doceavos")
        (
            r"\b((?i:el|del|al)\s+)([1-9]|[12]\d|3[01])/(0?[1-9]|1[0-2])(?![\d/])",
            lambda m: f"{m[1]}{m[2]} de {_MONTH_NAMES[int(m[3]) - 1]}",
        ),
        (
            r"(?<![\w/.,])(1[3-9]|2\d|3[01])/(0?[1-9]|1[0-2])(?![\d/])",
            lambda m: f"{m[1]} de {_MONTH_NAMES[int(m[2]) - 1]}",
        ),
        (r"(?<![\w/.,])(\d|10)/(10|5)(?![\w/]|[.,]\d)", r"\1 de \2"),  # ratings; NeMo: "ocho décimos"
        (
            r"(?<![\d.,])(\d+)\s*(?:vs\.?|VS\.?|v)\s*(\d+)(?![.,]?\d)",
            lambda m: f"{_spell(int(m[1]))} contra {_spell(int(m[2]))}",
        ),
        (r"(?<![\w.])[vV]s\.?(?=\s)", "contra"),  # not "VS Code"
        (r"(?<=\d)\s?[xX×]\s?(?=\d)", " por "),
        (r"(?<![\w.])x(?=\d)", "por "),  # "x2", "x1.5"
        (r"(?<=\d)\s?\+\s?(?=\d)", " más "),
        (r"(?<![\w+])\+(?=\d)", "más "),
        # scores, ranges and spans of years; NeMo reads "1-1" as a date and "1990-2000" as a phone number
        (rf"(?<![\w.,:/-]){_RANGE_END}\s?-\s?{_RANGE_END}(?![\w.,:/-]?\d|\w)", _range),
        (r"(?<=\d)-(?=[A-Za-z])", " "),  # "11-S", "0800-PELADO"
        (r"(?<![\w.,])[1-9]\d{0,2}(?: \d{3}){2,}(?![\w.,]?\d)", lambda m: m.group(0).replace(" ", "")),  # 1 000 000
        # phone numbers; NeMo glues the digits of "0800-333-1234" before a word ("trestrestres")
        (r"(?<![\w.,/:+-])(?:\+\d{1,3}[ -]?)?(?:\(?\d{1,4}\)?[ -]){0,3}\d{3,4}[ -]\d{4}(?![\w.,/:-]?\d|\w)", _phone),
        (r"(?<![\w.,/:+-])911(?![\w.,]?\d)", "nueve once"),
        (r"\b(RTX|GTX)\s?(\d\d)(?!00)(\d\d)\b", lambda m: f"{m[1]} {_spell(int(m[2]))} {_spell(int(m[3]))}"),
        (r"\b([Ss])alu2\b", r"\1aludos"),
        (r"(?<![\w.,])(\d+(?:[.,]\d+)?)\s?([GgMmTtKk][Bb])\b", _data),
        (r"(?<![\w.,])(\d+)(?:[.,](\d{1,3}))?\s?([kK]|M)\b(\s+de(?=\s+[a-záéíóúñ]))?", _kilo),
        (rf"(?<![\w.,])(-)?(\d+)\s?[º°](?:\s?([CF])(?![{_L}]))?", _degrees),
        (r"(?<![\w.,])(10|[1-9])\.?((?:er|ero|era|ro|ra|do|da|to|ta|vo|va|mo|ma|no|na)s?|[º°ª]s?)(?!\w)", _ordinal),
        (r"(?<![\w.,])\d+(?:\.\d+){2,}(?![\w.]?\d)", _version),
        # heights; NeMo reads "1.70" as "uno punto siete cero" and "1.50" as a time
        (
            rf"(?i)(?P<pre>\b(?:mid[eoa]n?|midió|midieron|mido|medí|medís|mides|medía|medían|medimos|medir|mediría"
            rf"|altura|estatura)\s+(?:[{_L}]+\s+)?)(?P<m>[12])[.,](?P<cm>\d\d)\b(?![.,]\d)(?:\s?(?:m|mts?|metros)\b)?",
            _height,
        ),
        (
            r"(?<![\w.,$])(?P<m>[12])[.,](?P<cm>\d\d)(?:\s?(?:m|mts?|metros)\b|(?=\s+de\s+(?:altura|alto|estatura)\b))",
            _height,
        ),
        (rf"(?<![\w.,$])(?P<m>1)\.(?P<cm>[4-9]\d)(?![\d.,]?\d|{_UNIT})", _bare_height),
        (rf"(?<![\w.,$])(?P<m>2)\.(?P<cm>[0-2]\d)(?![\d.,]?\d|{_UNIT})", _bare_height),
        (r"(?<![\w.,$])(\d+)\.(\d{1,2}|\d{4,})(?![\d.,]?\d)", _period_decimal),
        (r"(?<![\w.,])((?:1[89]|20)?[2-9]0|(?:1[89]|20)[0-9]0)'?s\b", r"\1"),  # "los 90s", "los 1980s"
        (r"(?<=\d)(?:st|nd|rd|th)\b", ""),
        (r"(\d:\d\d)\s?(?:hs|hrs?|h|horas)\b", r"\1"),  # "21:30 hs": the time says it already
        # NeMo reads "13:01" as "trece y un" before a word; the joiner marks the minutes for _POST
        (r"(?<![\d:])(\d{1,2}:[0-5]1)(?![\d:])(?!\s?(?:[ap]\.?\s?m\b|[AP]M\b))", rf"\1 {_WJ}"),
        (r"(?<=\d)\s?(hs|hrs?|segs?|mins|mts|kms|mg)\b", _abbr_unit),
        (
            r"(?<![\w.,])(\d{1,3}(?:\.\d{3})+|\d+)\s?(km/h|km²|km2|m²|m2|m³|m3|kph|mph|[Kk][mg]|KM|KG|cm|mm|ml"
            r"|ms|min|[ghlms])(?![\w²³/])",
            _measure,
        ),
        (rf"(?<!\S)\S*(?:\d|\w\.(?:{_EXTENSIONS})\b|\w@\w)\S*", _token),
        (r"(?<![\w.,:/$€£#+-])(?:\d{1,3}(?:\.\d{3})+|\d+)(?![\w%º°ª€$£]|\.?[ºª°]|[.,:/-]\d|\s?[€$£])", _integer),
        # Roman numerals in clear contexts
        (
            rf"\b((?i:{_ROMAN_NOUNS})\s+)({_ROMAN})\b((?:\s+(?:y|e|o|al|a|-)\s+(?:{_ROMAN})\b)*)",
            _roman_nouns,
        ),
        (r"\b((?i:los|mis|sus|tus)\s+|fiesta\s+de\s+)XV\b", r"\1quince"),  # "los XV de mi prima"
        (rf"\b([{_L}]+)\s+([IVXLCDM]+)\b", _regnal),
        (r"\b(del?)\s+([IVXLCDM]+)\s+(al?)\s+([IVXLCDM]+)\b", _span),
        (rf"\b([A-ZÁÉÍÓÚÑ][{_L}]*)\s+([IVXL]{{2,}})\b", _sequel),
        # shield the rest from NeMo's Roman numerals and abbreviations
        (rf"(?<![\w{_WJ}])([IVXLCDM]+|[SN])(?![\w{_WJ}])", _shield),
        (rf"(?<![\w{_WJ}])(?<!\d\s)([ivxlcdm]{{2,}})(?![\w{_WJ}])", rf"\1{_WJ}"),  # not units: "20 cm"
        (r"\b([Dd]a|d)(?=\.)", rf"\1{_WJ}"),
    ]
]

_FULL = {"un": "uno", "veintiún": "veintiuno"}
_NUM_END = "|".join(_ONES + _TENS + _HUNDREDS + ["cien", "mil", "millón", "millones"])
_CLAUSE_END = r"(?=\s*(?:[,;:!?]|\.(?!\.)|$)|\s+(?:de la|hs|horas?)\b)"
_POST: list[tuple[re.Pattern, str | Callable[[re.Match], str]]] = [
    (re.compile(p), r)
    for p, r in [
        (rf"\b(un|veintiún) ?{_WJ}", lambda m: _FULL[m[1]]),  # the minutes of "13:01 a casa"
        (rf" ?{_WJ}", ""),
        (r"\s+([.,;:!?…]+)(?=\s|$|['\"»)\]])", r"\1"),  # "Capítulo uno ." -> "Capítulo uno."
        (r"([¿¡(\[«])\s+", r"\1"),
        (r"\s+([)\]»])", r"\1"),
        # number words NeMo still apocopates in decimals and times: "tres coma un", "trece y un."
        (rf"\b((?:{_NUM_END}) coma (?:(?:{'|'.join(_TENS)}) y )?)(un|veintiún)\b", lambda m: m[1] + _FULL[m[2]]),
        (rf"\b((?:y|ciento|mil) )(un|veintiún)\b{_CLAUSE_END}", lambda m: m[1] + _FULL[m[2]]),
        (rf"\bveintiún\b{_CLAUSE_END}", "veintiuno"),
    ]
]


def _apply(rules: list, text: str) -> str:
    for pattern, repl in rules:
        text = pattern.sub(repl, text)
    return text


_ANNOTATION = re.compile(r"(<[^|>\n]+\|[^>\n]+>)")  # <word|pronunciation>, as in infer_v2_5


def normalize(text: str, nemo: Callable[[str], str]) -> str:
    """Normalize Spanish ``text``: the pre-rules, ``nemo`` (NeMo's normalize for "es"), the post-rules."""

    def run(t: str) -> str:
        return _apply(_POST, nemo(_apply(_PRE, t)))

    parts = _ANNOTATION.split(text)
    if len(parts) == 1:
        return run(text)
    # Pronunciation annotations pass through as written (NeMo read the M of "K IY1 M" as 1000).
    return "".join(
        p if i % 2 or not p.strip() else re.sub(r"\S(?:.*\S)?", lambda m: run(m.group(0)), p, flags=re.S)
        for i, p in enumerate(parts)
    )

"""Spanish rules around NeMo's text normalization, tuned for Argentine (Rioplatense) chat text.

``normalize(text, nemo)`` is ``post(nemo(pre(text)))``. The pre-rules handle what NeMo's
Spanish grammar reads wrong:

- Plain integers are spelled here, with agreement. NeMo always apocopates ("Capítulo un",
  "a la un", "ciento un personas") and misreads any number containing 800 before a word
  ("2.800 perros" -> "dos ochocientos perros").
- Money: ``$`` is pesos (the Latin American convention), ``US$``/``U$S``/``USD`` are dollars,
  cents go after "con": "$1.250,50" -> "mil doscientos cincuenta pesos con cincuenta".
- Heights ("mide 1,70" -> "uno setenta"), scores and ranges ("3-1" -> "tres a uno"; NeMo reads
  a date), "vs", "2x1", "24/7", ratings ("8/10"), decimals written with a period ("13.8").
- What NeMo leaves raw: GB/MB/TB, "k"/"M" multipliers, "hs", decades ("los 90s") and digits
  glued to letters ("MatiGamer99", "F1", "3D", "4K").
- Roman numerals are read only in clear contexts: after "siglo", "capítulo"..., after a regnal
  name ("Juan Pablo II" -> "segundo", "Luis XIV" -> "catorce": ordinals up to ten), in
  "del I al XII", and after a title ("Final Fantasy XIV" -> "catorce", two letters or more).
  Any other Roman-looking token ("MI", "CV", "GTA V", "vitamina D"), "S"/"N" (NeMo: "sur",
  "norte") and "da." ("doña") get an invisible word joiner, which no NeMo grammar matches and
  which is removed afterwards.

The post-rules remove the space NeMo leaves before punctuation and fix the number words it
still apocopates inside times and decimals ("trece y un ." -> "trece y uno.").
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
yo vos tú él ella ellos ellas nosotros me te se le les nos usted ustedes
no sí ya más menos muy también tampoco nomás solamente apenas casi ahora hoy ayer siempre nunca jamás
acá allá ahí aquí allí así todavía aún después antes luego recién igual nada algo todo
es son era eran fue fueron será serán sería sea sean está están estaba estaban estuvo hay había hubo habrá
tiene tienen tenía tenían tuvo tengo tenés tenemos va van iba iban vamos voy viene vienen vino sale salen
queda quedan pasa pasan dice dicen hace hacen puede pueden sigue siguen vale valen cuesta cuestan
soy sos eres somos estoy estás estamos
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
_FEM_NOUNS = frozenset(
    """
vez veces noche noches mujer mujeres foto fotos moto motos mano manos calle calles parte partes clase clases
red redes imagen imágenes ley leyes flor flores llave llaves torre torres tarde tardes muerte muertes
frase frases serie series radio radios piel pieles luz luces voz voces
""".split()
)
_MASC_A = frozenset(
    """
día días mapa mapas problema problemas tema temas sistema sistemas programa programas idioma idiomas clima
planeta planetas poema poemas drama dramas fantasma fantasmas pijama pijamas cometa cometas dilema dilemas
esquema esquemas síntoma síntomas trauma traumas enigma enigmas diagrama diagramas aroma aromas
giga gigas mega megas tera teras
""".split()
)
_FEM_END = re.compile(r"(?:a|as|ción|ciones|sión|siones|dad|dades|tud|tudes|umbre|umbres)$")
_MONTH_NAMES = "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split()
_MONTHS = "|".join(_MONTH_NAMES + ["setiembre"])
_UNIT = rf"\s?(?:km²|m²|m³|°|%|kph|mph|kg|km|cm|mm|ml|ms|min|am|pm|AM|PM|a\.m\.|p\.m\.|[ghlms])(?![{_L}])"
# Left to NeMo: "1 de mayo" (dates), "5 km", "3am" (measures, times).
_NEMO_AFTER = re.compile(rf"\s+de\s+(?i:{_MONTHS})\b|{_UNIT}")
_NEXT = re.compile(rf"\s+([{_L}]+)")
_COORD = re.compile(rf"\s+[oueya]\s+\d[\d.]*\s+([{_L}]+)")  # "1 o 2 veces", "de 1 a 3 horas"
_PREV = re.compile(rf"([{_L}]+)\s+$")


def _prev_word(text: str, pos: int) -> str:
    m = _PREV.search(text, 0, pos)
    return m.group(1).lower() if m else ""


def _is_fem(word: str) -> bool:
    w = word.lower()
    return w in _FEM_NOUNS or (w not in _MASC_A and not w.endswith(("ista", "istas")) and bool(_FEM_END.search(w)))


def _say(n: int, text: str, start: int, end: int) -> str:
    """Spell the number ``n`` found at ``text[start:end]`` the way its neighbors ask for."""
    nxt = _NEXT.match(text, end)
    word = nxt.group(1) if nxt else ""
    lw = word.lower()
    prev = _PREV.search(text, 0, start)
    pw = prev.group(1) if prev else ""
    head = text[: prev.start()].rstrip() if prev else ""
    # "Capítulo 1", "el Area 51 guardan": a label, not a count (a capital word is a name unless
    # it starts the sentence, or the whole text is shouted)
    name = pw[:1].isupper() and head and head[-1] not in ".!?¡¿…:;'\"«(" and not (len(word) > 1 and word.isupper())
    if pw.lower() in _LABELS or name:
        return _spell(n)
    if lw == "mil":  # "21 mil" -> "veintiún mil"; "200 mil personas" agrees with the noun after it
        noun = _NEXT.match(text, nxt.end())
        return " ".join(_apocope(_spell(n, "f" if noun and _is_fem(noun.group(1)) else "m").split()))
    if not word or lw in _CLOSED or (lw not in _NOUN_EN and _VERB.search(lw)):
        coord = _COORD.match(text, end)
        if (coord and _is_fem(coord.group(1))) or pw.lower() in ("la", "las"):
            return _spell(n, "f")  # "una o dos veces", "a la una", "a las veintiuna"
        return _spell(n)
    words = _spell(n, "f" if _is_fem(word) else "m")
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
    if "." not in s and (code := _code(digits)):
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
    n, cents, mult = int(m["n"].replace(".", "")), m["c"], m["mult"]
    if mult:  # "$1,5 millones": the decimals belong to the multiplier
        n, cents = _scaled(m["n"].replace(".", ""), cents, mult), None
    one, many = ("peso", "pesos") if m["cur"] == "$" else ("dólar", "dólares")
    c = int(cents.ljust(2, "0")) if cents else 0
    if n == 0 and c:
        return f"{_spell(c, 'm')} {'centavo' if c == 1 else 'centavos'}"
    words = _spell(n, "m")
    words += f"{' de' if words.endswith(_SCALES) else ''} {one if n == 1 else many}"
    return f"{words} con {_spell(c)}" if c else words


def _kilo(m: re.Match) -> str:
    """'2.5K' -> '2.500', spelled later with its context; '4K' and '8K' are resolutions."""
    whole, dec, mult, de = m.groups()
    if mult in "kK" and not dec and whole in ("4", "8"):
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
    "mts": "metro",
    "kms": "kilómetro",
}


def _abbr_unit(m: re.Match) -> str:
    one = re.search(r"(?<![\d.,])1$", m.string[: m.start()].rstrip())
    return f" {_ABBR_UNITS[m.group(1)]}{'' if one else 's'}"


def _period_decimal(m: re.Match) -> str:
    """'13.8' -> 'trece punto ocho' (NeMo: thousands, or a time); before a unit, a Spanish comma."""
    if _prev_word(m.string, m.start()) in ("la", "las"):
        return m.group(0)  # "a las 3.30": NeMo's time
    a, b = m.groups()
    if re.match(_UNIT, m.string[m.end() :]):
        return f"{a},{b}"
    return f"{_spell(int(a))} punto {_fraction(b)}"


def _bare_height(m: re.Match) -> str:
    """'1.80' alone, with a period, is a height in chat ('galanes de 1.80', '1.55 perfectos')."""
    return m.group(0) if _prev_word(m.string, m.start()) in ("la", "las") else _height(m)


_SUFFIXES = frozenset(
    "km kg cm mm ml ms m g h l min s mph kph am pm AM PM ro do to vo no mo er ra da ta va na ma".split()
)
_EXTENSIONS = "exe|com|net|org|ar|io|gg|tv|txt|pdf|jpg|png|gif|zip|rar|py|js"


def _token(m: re.Match) -> str:
    """Spell digits glued to letters ("MatiGamer99", "3D", "C5N"), except before the suffixes
    NeMo reads (units, am/pm, ordinals); "virus.exe" -> "virus punto exe". Mails and URLs as is."""
    tok = m.group(0)
    if re.search(r"@|://|www\.", tok):
        return tok
    tok = re.sub(rf"(?<=\w)\.(?=(?:{_EXTENSIONS})\b)", " punto ", tok)

    def glued(g: re.Match) -> str:
        digits, letters = g.groups()
        after_letter = g.start() > 0 and re.match(rf"[{_L}]", tok[g.start() - 1])
        if not after_letter and (not letters or letters in _SUFFIXES):
            return g.group(0)
        words = _code(digits) or _spell(int(digits))
        return f"{' ' if after_letter else ''}{words}{' ' + letters if letters else ''}"

    return re.sub(rf"(\d+)([{_L}]*)", glued, tok)


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


def _roman_nouns(m: re.Match) -> str:
    """'siglo XXI' -> 'siglo veintiuno', 'siglos XV y XVI' -> 'siglos quince y dieciséis'."""
    if not _roman(m.group(2)):
        return m.group(0)
    more = re.sub(_ROMAN, lambda r: _spell(_roman(r.group(0))) if _roman(r.group(0)) else r.group(0), m.group(3))
    return m.group(1) + _spell(_roman(m.group(2))) + more


def _regnal(m: re.Match) -> str:
    v = _roman(m.group(2))
    return f"{m.group(1)} {_ORDINALS[v - 1] if v <= 10 else _spell(v)}" if v and m.group(1) in _REGNAL else m.group(0)


def _span(m: re.Match) -> str:
    a, b = _roman(m.group(2)), _roman(m.group(4))
    return f"{m.group(1)} {_spell(a)} {m.group(3)} {_spell(b)}" if a and b else m.group(0)


def _sequel(m: re.Match) -> str:
    v = _roman(m.group(2))
    return f"{m.group(1)} {_spell(v)}" if v and m.group(1).lower() not in _CLOSED else m.group(0)


def _abbr(name: str) -> Callable[[re.Match], str]:
    # NeMo's own "EE. UU." -> "Estados Unidos" swallows the final period too; keep it at the end
    return lambda m: name + ("." if m.group(0).endswith(".") and m.end() == len(m.string) else "")


_RANGE_END = r"(0|[1-9]\d{0,2}|(?:1[5-9]|20)\d\d|[1-9]\d00)"  # up to 999, a year, or round hundreds
_ROMAN_NOUNS = "siglos?|capítulos?|tomos?|partes?|volumen|temporadas?|actos?|fases?|episodios?|libros?|niveles?"
_CURRENCY = r"US\$|U\$[SD]|USD|u\$s|\$"
_AMOUNT = r"(?P<n>\d{1,3}(?:\.\d{3})+|\d+)(?:[.,](?P<c>\d{1,2}))?(?!\d)(?:\s?(?P<mult>mil(?:lones|lón)?|[kKM])\b)?"
_PRE: list[tuple[re.Pattern, str | Callable[[re.Match], str]]] = [
    (re.compile(p), r)
    for p, r in [
        (r"\bRR\.?\s?HH\b\.?", _abbr("recursos humanos")),
        (r"\bEE\.?\s?UU\b\.?", _abbr("Estados Unidos")),
        (r"\bc/u\b", "cada uno"),
        (r"(?<![\w@])@(?=\w)", "arroba "),  # NeMo spells a handle letter by letter, and its comma as "coma"
        (r"\b[Nn](?:ro\.?|\.?\s?[º°]|\.)\s?(?=\d)|\bnúm\.\s?(?=\d)|#\s?(?=\d)", "número "),
        (rf"(?<![\w.,/:-]){_AMOUNT}\s?(?P<cur>US\$|U\$[SD]|USD|u\$s)(?!\w)", _money),
        (rf"(?<![\w$])(?P<cur>{_CURRENCY})\s?{_AMOUNT}(?![/:-]\d)", _money),
        (r"\b24/7\b", "veinticuatro siete"),
        # "el 31/12": NeMo reads a day and month without a year as a fraction ("treinta y un doceavos")
        (
            r"\b((?i:el|del|al)\s+)([1-9]|[12]\d|3[01])/(0?[1-9]|1[0-2])(?![\d/])",
            lambda m: f"{m[1]}{m[2]} de {_MONTH_NAMES[int(m[3]) - 1]}",
        ),
        (r"(?<![\w/.,])(\d|10)/10(?![\w/]|[.,]\d)", r"\1 de 10"),  # ratings; NeMo: "ocho décimos"
        (
            r"(?<![\d.,])(\d+)\s*(?:vs\.?|VS\.?|v)\s*(\d+)(?![.,]?\d)",
            lambda m: f"{_spell(int(m[1]))} contra {_spell(int(m[2]))}",
        ),
        (r"(?i)(?<![\w.])vs\.?(?=\s)", "contra"),
        (r"(?<=\d)\s?[xX×]\s?(?=\d)", " por "),
        (r"(?<=\d)\s?\+\s?(?=\d)", " más "),
        (r"(?<![\w+])\+(?=\d)", "más "),
        # scores, ranges and spans of years; NeMo reads "1-1" as a date and "1990-2000" as a phone number
        (rf"(?<![\w.,:/-]){_RANGE_END}\s?-\s?{_RANGE_END}(?![\w.,:/-]?\d|\w)", r"\1 a \2"),
        (r"(?<=\d)-(?=[A-Za-z])", " "),  # "11-S", "0800-PELADO"
        # phone numbers; NeMo glues the digits of "0800-333-1234" before a word ("trestrestres")
        (r"(?<![\w.,/:+-])(?:\+\d{1,3}[ -]?)?(?:\(?\d{1,4}\)?[ -]){0,3}\d{3,4}[ -]\d{4}(?![\w.,/:-]?\d|\w)", _phone),
        (r"(?<![\w.,])(\d+(?:[.,]\d+)?)\s?([GgMmTtKk][Bb])\b", _data),
        (r"(?<![\w.,])(\d+)(?:[.,](\d{1,3}))?\s?([kK]|M)\b(\s+de(?=\s+[a-záéíóúñ]))?", _kilo),
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
        (r"(\d:\d\d)\s?(?:hs|hrs?|h)\b", r"\1"),  # "21:30 hs": the time says it already
        (r"(?<=\d)\s?(hs|hrs?|segs?|mins|mts|kms)\b", _abbr_unit),
        (rf"\S*(?:\d|\w\.(?:{_EXTENSIONS})\b)\S*", _token),
        (r"(?<![\w.,:/$€£#+-])(?:\d{1,3}(?:\.\d{3})+|\d+)(?![\w%º°ª€$£]|\.?[ºª°]|[.,:/-]\d|\s?[€$£])", _integer),
        # Roman numerals in clear contexts
        (
            rf"\b((?i:{_ROMAN_NOUNS})\s+)({_ROMAN})\b((?:\s+(?:y|e|o|al|a|-)\s+(?:{_ROMAN})\b)*)",
            _roman_nouns,
        ),
        (rf"\b([{_L}]+)\s+([IVXLCDM]+)\b", _regnal),
        (r"\b(del?)\s+([IVXLCDM]+)\s+(al?)\s+([IVXLCDM]+)\b", _span),
        (rf"\b([A-ZÁÉÍÓÚÑ][{_L}]*)\s+([IVX]{{2,}})\b", _sequel),
        # shield the rest from NeMo's Roman numerals and abbreviations
        (rf"(?<![\w{_WJ}])([IVXLCDM]+|[SN])(?![\w{_WJ}])", rf"\1{_WJ}"),
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
        return _apply(_POST, nemo(_apply(_PRE, t)).replace(_WJ, ""))

    parts = _ANNOTATION.split(text)
    if len(parts) == 1:
        return run(text)
    # Pronunciation annotations pass through as written (NeMo read the M of "K IY1 M" as 1000).
    return "".join(
        p if i % 2 or not p.strip() else re.sub(r"\S(?:.*\S)?", lambda m: run(m.group(0)), p, flags=re.S)
        for i, p in enumerate(parts)
    )

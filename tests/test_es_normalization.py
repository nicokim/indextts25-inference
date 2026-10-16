"""Spanish text normalization: NeMo plus the rules in indextts25/utils/es_rules.py.

CPU only. The first run compiles NeMo's Spanish grammar (about 25 s); later runs load it from
the cache in under a second. Every case also checks that normalizing the output again changes
nothing, because synthesize() may receive text that was already normalized.
"""

import pytest

pytest.importorskip("nemo_text_processing")

from indextts25.utils.es_rules import _spell  # noqa: E402
from indextts25.utils.nemo_tn import normalize_text  # noqa: E402


def check(text, expected):
    out = normalize_text(text, "es")
    assert out == expected
    assert normalize_text(out, "es") == out


@pytest.mark.parametrize(
    "n,full,masc,fem",
    [
        (0, "cero", "cero", "cero"),
        (1, "uno", "un", "una"),
        (16, "dieciséis", "dieciséis", "dieciséis"),
        (21, "veintiuno", "veintiún", "veintiuna"),
        (31, "treinta y uno", "treinta y un", "treinta y una"),
        (100, "cien", "cien", "cien"),
        (101, "ciento uno", "ciento un", "ciento una"),
        (500, "quinientos", "quinientos", "quinientas"),
        (1001, "mil uno", "mil un", "mil una"),
        (2001, "dos mil uno", "dos mil un", "dos mil una"),
        (21000, "veintiún mil", "veintiún mil", "veintiún mil"),
        (200000, "doscientos mil", "doscientos mil", "doscientas mil"),
        (1_000_000, "un millón", "un millón", "un millón"),
        (1_500_000, "un millón quinientos mil", "un millón quinientos mil", "un millón quinientas mil"),
        (200_000_000, "doscientos millones", "doscientos millones", "doscientos millones"),
        (10**9, "mil millones", "mil millones", "mil millones"),
        (10**12, "un billón", "un billón", "un billón"),
    ],
)
def test_spell(n, full, masc, fem):
    assert (_spell(n), _spell(n, "m"), _spell(n, "f")) == (full, masc, fem)


# NeMo apocopates every number ending in 1 and never agrees with feminine nouns.
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Capítulo 1.", "Capítulo uno."),
        ("Abrí el libro en la página 21.", "Abrí el libro en la página veintiuno."),
        ("El Área 51 es grande.", "El Área cincuenta y uno es grande."),
        ("En el Area 51 guardan cosas.", "En el Area cincuenta y uno guardan cosas."),
        ("Es un juego, el número 1 tiene uniforme.", "Es un juego, el número uno tiene uniforme."),
        ("Person 1 tiene el frisbee.", "Person uno tiene el frisbee."),
        ("Desde el 2001.", "Desde el dos mil uno."),
        ("Tengo 3 años, nací en 2021.", "Tengo tres años, nací en dos mil veintiuno."),
        ("Ganó 3 a 1.", "Ganó tres a uno."),
        ("Nos vemos a la 1.", "Nos vemos a la una."),
        ("Arrancamos a las 21.", "Arrancamos a las veintiuna."),
        ("Tengo 21 pero me siento de 80.", "Tengo veintiuno pero me siento de ochenta."),
        ("Tengo 1 perro.", "Tengo un perro."),
        ("Mandame 1 bit nomás.", "Mandame un bit nomás."),
        ("Faltan 31 días.", "Faltan treinta y un días."),
        ("Somos 101 personas.", "Somos ciento una personas."),
        ("Tengo 500 horas en ranked.", "Tengo quinientas horas en ranked."),
        ("Libros de 500 páginas.", "Libros de quinientas páginas."),
        ("Después de 500 partidas.", "Después de quinientas partidas."),
        ("Se sacan 500 fotos.", "Se sacan quinientas fotos."),
        ("Es omakase de 200 lucas.", "Es omakase de doscientas lucas."),
        ("Se mueren 1500 personas.", "Se mueren mil quinientas personas."),
        ("Vinieron 200.000 personas.", "Vinieron doscientas mil personas."),
        ("Somos 21 mil personas.", "Somos veintiún mil personas."),
        ("Lo hice 1 o 2 veces.", "Lo hice una o dos veces."),
        ("Hay 1.000.000 de razones.", "Hay un millón de razones."),
        ("Se ven '06', '07' y '08'.", "Se ven 'cero seis', 'cero siete' y 'cero ocho'."),
        ("25, 30 con toda la furia.", "veinticinco, treinta con toda la furia."),
        ("Usá el código 007.", "Usá el código cero cero siete."),
        ("Llamá al 1123456789.", "Llamá al uno uno dos tres cuatro cinco seis siete ocho nueve."),
        ("Son 1000000 de pesos.", "Son un millón de pesos."),
    ],
)
def test_integers(text, expected):
    check(text, expected)


# NeMo read any written number containing 800 as "ochocientos" plus loose digits before a word.
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Son 2.800 perros.", "Son dos mil ochocientos perros."),
        ("Son 8000 perros.", "Son ocho mil perros."),
        ("Son 8005 perros.", "Son ocho mil cinco perros."),
        ("Hay 18000 perros.", "Hay dieciocho mil perros."),
        ("Hay 100.800 perros.", "Hay cien mil ochocientos perros."),
        ("Hay 800.000 perros.", "Hay ochocientos mil perros."),
        ("Hay 800 perros.", "Hay ochocientos perros."),
        ("Se quedaron en 1800 y siguen.", "Se quedaron en mil ochocientos y siguen."),
        (
            "El 0800-333-1234 no atiende nunca.",
            "El cero ochocientos tres tres tres uno dos tres cuatro no atiende nunca.",
        ),
        ("Kimox es 0800-PELADO.", "Kimox es cero ochocientos PELADO."),
    ],
)
def test_eight_hundred(text, expected):
    check(text, expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Costó $1.250,50.", "Costó mil doscientos cincuenta pesos con cincuenta."),
        ("El alfajor sale $2.300.", "El alfajor sale dos mil trescientos pesos."),
        ("Bits de $1.", "Bits de un peso."),
        ("Cuesta $0,50.", "Cuesta cincuenta centavos."),
        ("Pagué $1.50 por eso.", "Pagué un peso con cincuenta por eso."),
        ("Hay $1.000.000 en juego.", "Hay un millón de pesos en juego."),
        ("Son $2 millones.", "Son dos millones de pesos."),
        ("Son $1,5 millones.", "Son un millón quinientos mil pesos."),
        ("Pagué US$100.", "Pagué cien dólares."),
        ("Son U$S 1.500.", "Son mil quinientos dólares."),
        ("Son USD 21.", "Son veintiún dólares."),
        ("Cuesta 20 USD.", "Cuesta veinte dólares."),
        ("Son 20 USD el 12/03.", "Son veinte dólares el doce de marzo."),
        ("Salen 100 pesos c/u.", "Salen cien pesos cada uno."),
    ],
)
def test_money(text, expected):
    check(text, expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Mide 1.70 el enano.", "Mide uno setenta el enano."),
        ("Midió 1.70 en la revisión.", "Midió uno setenta en la revisión."),
        ("Medí 1.50 y me las banco.", "Medí uno cincuenta y me las banco."),
        ("Debe medir como 1.70 el enano.", "Debe medir como uno setenta el enano."),
        ("Mide 1,70 m.", "Mide uno setenta."),
        ("1.50 de altura y ganas.", "uno cincuenta de altura y ganas."),
        ("Tienen galanes de 1.80 con piel.", "Tienen galanes de uno ochenta con piel."),
        ("1.55 perfectos.", "uno cincuenta y cinco perfectos."),
        ("Juega uno de 2.05 m.", "Juega uno de dos cero cinco."),
        # decimals with a period: NeMo read thousands or a time
        ("Counter 1.6 pirata.", "Counter uno punto seis pirata."),
        ("Una versión 2.0 con menos bugs.", "Una versión dos punto cero con menos bugs."),
        ("Hace 13.8 billones de años.", "Hace trece punto ocho billones de años."),
        ("Comprás a 0.0001, y vendés.", "Comprás a cero punto cero cero cero uno, y vendés."),
        ("Recorrí 15.5 km.", "Recorrí quince coma cinco kilómetros."),
        ("Nos vemos a las 3.30.", "Nos vemos a las tres y media."),
    ],
)
def test_heights_and_decimals(text, expected):
    check(text, expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Van 1-1.", "Van uno a uno."),
        ("Perdieron 7-1 contra Alemania.", "Perdieron siete a uno contra Alemania."),
        ("Empataron 0-0.", "Empataron cero a cero."),
        ("Mata en 3-5 días.", "Mata en tres a cinco días."),
        ("Tiene unos 35-40 años.", "Tiene unos treinta y cinco a cuarenta años."),
        ("Entre 2500-3000 pesos.", "Entre dos mil quinientos a tres mil pesos."),
        ("Del 1990-2000.", "Del mil novecientos noventa a dos mil."),
        ("Juego 1 vs 1.", "Juego uno contra uno."),
        ("Un 1v1 rápido.", "Un uno contra uno rápido."),
        ("Messi vs Cristiano.", "Messi contra Cristiano."),
        ("Un departamento de 2x2.", "Un departamento de dos por dos."),
        ("Rompe las bolas 24/7.", "Rompe las bolas veinticuatro siete."),
        ("Le doy un 8/10.", "Le doy un ocho de diez."),
        ("Feedeo 0/10.", "Feedeo cero de diez."),
        ("Soy el #1.", "Soy el número uno."),
        ("Vivo en el N.º 5.", "Vivo en el número cinco."),
        ("Hay +50 bots.", "Hay más cincuenta bots."),
        ("Sabe que 2+2 es 4.", "Sabe que dos más dos es cuatro."),
        # still NeMo's, untouched
        ("Llamá al 4567-8901.", "Llamá al cuatro cinco seis siete ocho nueve cero uno."),
        ("Nos vemos el 12/03/2026.", "Nos vemos el doce de marzo de dos mil veintiséis."),
        ("El 1 de mayo no se labura.", "El primero de mayo no se labura."),
        ("El 1.º de mayo.", "El primero de mayo."),
        ("Nos vemos el 31/12.", "Nos vemos el treinta y uno de diciembre."),
        ("El 5/10 nos vemos.", "El cinco de octubre nos vemos."),
        ("Hay 1/2 kilo.", "Hay medio kilo."),
        ("Llamá al 11 4567-8901.", "Llamá al once cuatro cinco seis siete ocho nueve cero uno."),
        (
            "Llamá al +54 9 11 4567-8901.",
            "Llamá al más cincuenta y cuatro nueve once cuatro cinco seis siete ocho nueve cero uno.",
        ),
        ("Llamá al 333-1234.", "Llamá al tres tres tres uno dos tres cuatro."),
    ],
)
def test_scores_ranges_and_signs(text, expected):
    check(text, expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Pesa 1 GB.", "Pesa un giga."),
        ("Son 500 MB y 2 TB.", "Son quinientos megas y dos teras."),
        ("El juego pesa 50 GB.", "El juego pesa cincuenta gigas."),
        ("Tenés que descargar 50GB.", "Tenés que descargar cincuenta gigas."),
        ("Bajate 500gb de porno.", "Bajate quinientos gigas de porno."),
        ("Mejoró de 1kb a 2kb?", "Mejoró de un kilobyte a dos kilobytes?"),
        ("Tengo 100k.", "Tengo cien mil."),
        ("Tengo 1k seguidores.", "Tengo mil seguidores."),
        ("Tengo 12K subs.", "Tengo doce mil subs."),
        ("Tengo 2.5K, más que vos.", "Tengo dos mil quinientos, más que vos."),
        ("Un boludo con 100k de seguidores.", "Un boludo con cien mil seguidores."),
        ("Ganó 100M de dólares.", "Ganó cien millones de dólares."),
        ("Son 5M seguidores.", "Son cinco millones de seguidores."),
        ("Porno en 4K.", "Porno en cuatro K."),
        ("Lo veo en 4k.", "Lo veo en cuatro K."),
        ("A las 21 hs.", "A las veintiuna horas."),
        ("A las 21:30 hs.", "A las veintiuna y media."),
        ("Esperá 30 seg.", "Esperá treinta segundos."),
        ("Es de los 40s o 50s.", "Es de los cuarenta o cincuenta."),
        ("Hentai de los 90s.", "Hentai de los noventa."),
        ("Corrí 5 km y tardé 45 min.", "Corrí cinco kilómetros y tardé cuarenta y cinco minutos."),
        ("Son las 13:01.", "Son las trece y uno."),
        ("A la 1:01 de la tarde.", "A la una y uno de la tarde."),
        ("Ahora 1,1 km.", "Ahora uno coma uno kilómetros."),
        ("Son 2,31 km.", "Son dos coma treinta y uno kilómetros."),
        ("A las 12:21.", "A las doce y veintiuno."),
    ],
)
def test_units(text, expected):
    check(text, expected)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Gracias MatiGamer99.", "Gracias MatiGamer noventa y nueve."),
        ("Gracias FacuGamer21?", "Gracias FacuGamer veintiuno?"),
        ("PibeNewbie22 100 bits nada más?", "PibeNewbie veintidós cien bits nada más?"),
        ("Piloto de F1.", "Piloto de F uno."),
        ("Mi modelo 3D.", "Mi modelo tres D."),
        ("Le dio 3 D.", "Le dio tres D."),
        ("Rompé el WPA2.", "Rompé el WPA dos."),
        ("Calidad 480p.", "Calidad cuatrocientos ochenta p."),
        ("El 5G te fríe.", "El cinco G te fríe."),
        ("Se pajean con C5N.", "Se pajean con C cinco N."),
        ("Más caca que H2O.", "Más caca que H dos O."),
        ("Intel i9, pero no.", "Intel i nueve, pero no."),
        ("Como un abandona2.", "Como un abandona dos."),
        ("El 11S es eso.", "El once S es eso."),
        ("El 11 S fue eso.", "El once S fue eso."),
        ("El 11-S fue eso.", "El once S fue eso."),
        ("Vive en la 32nd Street.", "Vive en la treinta y dos Street."),
        ("Despertate a las 3am.", "Despertate a las tres de la mañana."),
        ("El vecino del 4to.", "El vecino del cuarto."),
        ("Necesitás un AK-47, no una guitarra.", "Necesitás un AK cuarenta y siete, no una guitarra."),
    ],
)
def test_glued_digits(text, expected):
    check(text, expected)


# NeMo read capital letters as Roman numerals and a few words as abbreviations.
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Es MI fantasía.", "Es MI fantasía."),
        ("I hope you die.", "I hope you die."),
        ("Mandale tu CV.", "Mandale tu CV."),
        ("Nada de talle L.", "Nada de talle L."),
        ("Te falta vitamina D.", "Te falta vitamina D."),
        ("Necesitaban vitamina D urgente.", "Necesitaban vitamina D urgente."),
        ("Con rayos X truchos.", "Con rayos X truchos."),
        ("Tenés hepatitis C, che.", "Tenés hepatitis C, che."),
        ("Decímelo por MD.", "Decímelo por MD."),
        ("GTA V.", "GTA V."),
        ("Escribí aprobado con V.", "Escribí aprobado con V."),
        ("C o C++ para hacerte el hacker.", "C o C++ para hacerte el hacker."),
        ("Un mix raro.", "Un mix raro."),
        ("El talle S, che.", "El talle S, che."),
        ("Qué más da.", "Qué más da."),
        ("Mi Instagram es @KohAIoficial, pero no.", "Mi Instagram es arroba KohAIoficial, pero no."),
        ("El de Kimox es @calvocaliente69.", "El de Kimox es arroba calvocaliente sesenta y nueve."),
        ("Mandá fotos a RR.HH., el acoso funciona.", "Mandá fotos a recursos humanos, el acoso funciona."),
        ("A los forros de RRHH les encanta.", "A los forros de recursos humanos les encanta."),
        ("Viven en EE.UU.", "Viven en Estados Unidos."),
        ("Ponele 'not a virus.exe'.", "Ponele 'not a virus punto exe'."),
        ("Mandales un .exe porno.", "Mandales un .exe porno."),
        # "coma" is also a verb, and "un..." trails off: neither is a number
        ("Que se coma un asado.", "Que se coma un asado."),
        ("Sos un... no sé.", "Sos un... no sé."),
        ("Un femboy vs una mina.", "Un femboy contra una mina."),
        # Roman numerals in clear contexts
        ("Es el petróleo del siglo XXI.", "Es el petróleo del siglo veintiuno."),
        ("Viven en el siglo XV.", "Viven en el siglo quince."),
        ("Entre los siglos XV y XVI.", "Entre los siglos quince y dieciséis."),
        ("Leé el capítulo IV.", "Leé el capítulo cuatro."),
        ("Juan Pablo II era polaco.", "Juan Pablo segundo era polaco."),
        ("Luis XIV fue rey.", "Luis catorce fue rey."),
        ("Felipe VI y Carlos V.", "Felipe sexto y Carlos quinto."),
        ("Final Fantasy XIV está lleno.", "Final Fantasy catorce está lleno."),
        ("Son del I al XII, por si no fuiste.", "Son del uno al doce, por si no fuiste."),
        ("NUNCA VI ALGO ASÍ.", "NUNCA VI ALGO ASÍ."),
    ],
)
def test_capitals(text, expected):
    check(text, expected)


def test_pronunciation_annotations_pass_through():
    # NeMo read the M of the annotation as 1000 and its S as "sur".
    check("Dale <Kimox|K IY1 M AA0 K S> 2.", "Dale <Kimox|K IY1 M AA0 K S> dos.")


# Number-heavy sentences with their spoken form, corrected by hand.
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Costó $1.250,50.", "Costó mil doscientos cincuenta pesos con cincuenta."),
        (
            "El alfajor sale $2.300 y el café $1.800.",
            "El alfajor sale dos mil trescientos pesos y el café mil ochocientos pesos.",
        ),
        ("Me cobraron $15.000 por cortarme el pelo.", "Me cobraron quince mil pesos por cortarme el pelo."),
        (
            "El dólar cerró a $1.480 y yo sigo pobre.",
            "El dólar cerró a mil cuatrocientos ochenta pesos y yo sigo pobre.",
        ),
        ("Son 3 cuotas de $999 sin interés.", "Son tres cuotas de novecientos noventa y nueve pesos sin interés."),
        ("Arranco el stream a las 21:30.", "Arranco el stream a las veintiuna y media."),
        ("Me levanté a las 7:15 y ya estaba cansada.", "Me levanté a las siete y cuarto y ya estaba cansada."),
        ("Son las 14:30.", "Son las catorce y media."),
        (
            "El 25 de mayo de 1810 fue la revolución.",
            "El veinticinco de mayo de mil ochocientos diez fue la revolución.",
        ),
        ("Nos vemos el 12/03/2026 en el stream.", "Nos vemos el doce de marzo de dos mil veintiséis en el stream."),
        ("El 1 de mayo no se labura.", "El primero de mayo no se labura."),
        ("En 1986 ganamos el Mundial.", "En mil novecientos ochenta y seis ganamos el Mundial."),
        ("Desde el 2001 que no veo un quilombo así.", "Desde el dos mil uno que no veo un quilombo así."),
        ("El 2026 va a ser mi año.", "El dos mil veintiséis va a ser mi año."),
        ("La inflación fue del 3,5 % este mes.", "La inflación fue del tres coma cinco por ciento este mes."),
        ("Me queda el 15% de batería.", "Me queda el quince por ciento de batería."),
        ("El 100% de mi chat son bots.", "El cien por ciento de mi chat son bots."),
        ("Llamá al 4567-8901.", "Llamá al cuatro cinco seis siete ocho nueve cero uno."),
        (
            "El 0800-333-1234 no atiende nunca.",
            "El cero ochocientos tres tres tres uno dos tres cuatro no atiende nunca.",
        ),
        ("Vivo en el 2º piso.", "Vivo en el segundo piso."),
        ("Quedé 3º en el torneo.", "Quedé tercero en el torneo."),
        ("Es la 5ª vez que te lo digo.", "Es la quinta vez que te lo digo."),
        ("Hoy leemos el capítulo 1.", "Hoy leemos el capítulo uno."),
        ("Abrí el libro en la página 21.", "Abrí el libro en la página veintiuno."),
        ("Mide 1,70 y se cree basquetbolista.", "Mide uno setenta y se cree basquetbolista."),
        ("Con 1.85 de altura no pasa por la puerta.", "Con uno ochenta y cinco de altura no pasa por la puerta."),
        ("Ganamos 3 a 1 con un gol en contra.", "Ganamos tres a uno con un gol en contra."),
        ("Perdimos 2-0 contra unos pibes de 12.", "Perdimos dos a cero contra unos pibes de doce."),
        ("El juego pesa 50 GB y mi disco tiene 20.", "El juego pesa cincuenta gigas y mi disco tiene veinte."),
        ("Corrí 5 km y tardé 45 min.", "Corrí cinco kilómetros y tardé cuarenta y cinco minutos."),
        ("Vení, 1 vs 1 si te la bancás.", "Vení, uno contra uno si te la bancás."),
        ("Tengo 1 perro y 21 gatos.", "Tengo un perro y veintiún gatos."),
        ("Cumplo 31 años y somos 21 en la fiesta.", "Cumplo treinta y un años y somos veintiuno en la fiesta."),
        ("Vinieron 101 personas a la 1 de la mañana.", "Vinieron ciento una personas a la una de la mañana."),
        ("Tengo 2.500 seguidores y 10.000 bits.", "Tengo dos mil quinientos seguidores y diez mil bits."),
    ],
)
def test_hand_corrected(text, expected):
    check(text, expected)

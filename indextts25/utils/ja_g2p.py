import random
import re


class JapaneseG2PProcessor:
    """
    Japanese text segmentation + hiragana conversion processor.
    Requires fugashi + unidic-lite (or a system MeCab).
    Install: pip install fugashi unidic-lite
    """

    def __init__(self, g2p_ratio=0.2):
        self.g2p_ratio = g2p_ratio
        self._init_tagger()

    def _init_tagger(self):
        try:
            import fugashi
            self.tagger = fugashi.Tagger()
            self.backend = 'fugashi'
        except ImportError:
            try:
                import MeCab
                self.tagger = MeCab.Tagger('-Ochasen')
                self.backend = 'mecab'
            except ImportError:
                raise ImportError(
                    "Please install fugashi: pip install fugashi unidic-lite, "
                    "or install a system MeCab and then pip install mecab-python3"
                )

    def tokenize(self, text: str) -> list:
        """
        Tokenize Japanese text; returns a list [(surface, reading_katakana), ...].
        reading is the katakana pronunciation; it equals surface if unavailable.
        """
        tokens = []
        if self.backend == 'fugashi':
            for token in self.tagger(text):
                surface = token.surface
                try:
                    reading = token.feature.kana
                    if not reading or reading == '*':
                        reading = surface
                except AttributeError:
                    feat = token.feature.split(',')
                    reading = feat[7] if len(feat) > 7 and feat[7] != '*' else surface
                tokens.append((surface, reading))
        else:
            for line in self.tagger.parse(text).splitlines():
                if line in ('EOS', ''):
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    surface = parts[0]
                    reading = parts[1] if parts[1] != '*' else parts[0]
                    tokens.append((surface, reading))
        return tokens

    @staticmethod
    def kata2hira(text: str) -> str:
        """Katakana → hiragana (ァ-ン → ぁ-ん)"""
        return ''.join(
            chr(ord(ch) - 0x60) if 0x30A1 <= ord(ch) <= 0x30F6 else ch
            for ch in text
        )

    @staticmethod
    def _has_kanji(text: str) -> bool:
        """Whether the string contains any kanji (CJK ideograph)"""
        return any('\u4e00' <= ch <= '\u9fff' for ch in text)

    def _process_segment(self, text: str) -> str:
        """Partially replace kanji tokens with hiragana in a single space-free segment."""
        tokens = self.tokenize(text)
        kanji_indices = [i for i, (surface, _) in enumerate(tokens) if self._has_kanji(surface)]
        num_to_replace = int(len(kanji_indices) * self.g2p_ratio)
        if num_to_replace == 0 and kanji_indices and random.random() < self.g2p_ratio:
            num_to_replace = 1
        replace_set = set(random.sample(kanji_indices, min(num_to_replace, len(kanji_indices))))
        result = []
        for i, (surface, reading) in enumerate(tokens):
            if i in replace_set:
                hira = self.kata2hira(reading)
                result.append(hira)
            else:
                result.append(surface)
        return ' '.join(result)


    def process_ja_text(self, text: str) -> str:
        """
        Tokenize Japanese text, then replace tokens containing kanji with their
        hiragana reading with probability g2p_ratio; the rest keep the original text.
        Spaces present in the input are preserved at the same positions in the output.
        """
        # Split on spaces keeping their positions, process each segment, then join back
        parts = re.split(r'( +)', text)   # odd indices are spaces, even ones text segments
        return ''.join(
            self._process_segment(p) if p.strip() else p
            for p in parts
        )

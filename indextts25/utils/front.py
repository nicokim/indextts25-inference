import logging
import os
import re
from functools import lru_cache

logger = logging.getLogger(__name__)


class TextNormalizer:
    def __init__(self, enable_glossary=False):
        self.zh_normalizer = None
        self.en_normalizer = None
        self._zh_en_error = None
        self.char_rep_map = {
            "：": ",",
            "；": ",",
            ";": ",",
            "，": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "\n": " ",
            "·": "-",
            "、": ",",
            "...": "…",
            ",,,": "…",
            "，，，": "…",
            "……": "…",
            "“": "'",
            "”": "'",
            '"': "'",
            "‘": "'",
            "’": "'",
            "（": "'",
            "）": "'",
            "(": "'",
            ")": "'",
            "《": "'",
            "》": "'",
            "【": "'",
            "】": "'",
            "[": "'",
            "]": "'",
            "—": "-",
            "～": "-",
            "~": "-",
            "「": "'",
            "」": "'",
            ":": ",",
        }
        self.zh_char_rep_map = {
            "$": ".",
            **self.char_rep_map,
        }
        self.clean_pattern = re.compile("|".join(re.escape(p) for p in self.char_rep_map.keys()))
        self.enable_glossary = enable_glossary
        # Term glossary: lets the user define how technical terms are read aloud
        # Format: {"original term": {"en": "English reading", "zh": "Chinese reading"}}
        # "M.2": {"en": "M dot two", "zh": "M 二"},
        # "PCIe 5.0": {"en": "PCIE five", "zh": "PCIE 五点零"},
        # "PCIe 4.0": {"en": "PCIE four", "zh": "PCIE 四点零"},
        # "AHCI": "A H C I",
        # "TTS": "T T S",
        # "Inc.": {"en": "Ink"},
        # ".json": {"en": " dot Jay-Son", "zh": "点 Jay-Son"},
        # "C++": {"en": "C plus plus", "zh": "C 加加"},
        # "C#": "C sharp"
        # self.term_glossary = {
        #     "C++": {"en": "C plus plus", "zh": "C 加加"},
        #     "C#": "C sharp",
        #     "CMake": "C Make",
        # }
        self.term_glossary = dict()

    def match_email(self, email):
        # Regex for an email address: alphanumeric@alphanumeric.letters
        pattern = r"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$"
        return re.match(pattern, email) is not None

    PINYIN_TONE_PATTERN = r"(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])"
    """
    Matches pinyin with a tone number: pinyin+digit, tones 1-5, where 5 is the neutral tone
    e.g. xuan4, jve2, ying1, zhong4, shang5
    Does not match: beta1, voice2
    """
    NAME_PATTERN = r"[\u4e00-\u9fff]+(?:[-·—][\u4e00-\u9fff]+){1,2}"
    """
    Matches person names written as Chinese·Chinese or Chinese·Chinese-Chinese
    e.g. 克里斯托弗·诺兰, 约瑟夫·高登-莱维特 (Christopher Nolan, Joseph Gordon-Levitt)
    """

    TECH_TERM_PATTERN = r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+"
    """
    Matches technical terms, format: a letter + (letters or digits)* + (-letters or digits)+
    e.g. GPT-5-nano, F5-TTS, Fish-Speech, GPT-5, CosyVoice-2
    Must start with a letter so pure numbers don't match (e.g. phone number 135-4567-8900)
    Protects the hyphens so the Chinese normalizer doesn't read them as minus signs (e.g. "负五减", i.e. "negative five minus")
    """

    # Matches common English 's contractions, only to replace them with "is"; does not match every 's
    ENGLISH_CONTRACTION_PATTERN = r"(what|where|who|which|how|t?here|it|s?he|that|this)'s"


    def use_chinese(self, s):
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", s))
        has_alpha = bool(re.search(r"[a-zA-Z]", s))
        is_email = self.match_email(s)
        if has_chinese or not has_alpha or is_email:
            return True

        has_pinyin = bool(re.search(TextNormalizer.PINYIN_TONE_PATTERN, s, re.IGNORECASE))
        return has_pinyin

    def load(self) -> bool:
        """Loads the WeTextProcessing zh/en normalizers (extra [zh]) the first time they
        are needed. Returns whether they are available; without them, en/zh go via NeMo."""
        if self.zh_normalizer is None and self._zh_en_error is None:
            try:
                self._load_zh_en()
            except (ImportError, OSError) as e:
                self.zh_normalizer = self.en_normalizer = None
                self._zh_en_error = e
        return self.zh_normalizer is not None

    def _load_zh_en(self):
        import platform

        if platform.system() != "Linux":  # Mac and Windows
            from wetext import Normalizer

            self.zh_normalizer = Normalizer(remove_erhua=False, lang="zh", operator="tn")
            self.en_normalizer = Normalizer(lang="en", operator="tn")
        else:
            from tn.chinese.normalizer import Normalizer as NormalizerZh
            from tn.english.normalizer import Normalizer as NormalizerEn
            # wetext, like NeMo, leaves its logger at INFO with its own handler: loads print to stderr.
            logging.getLogger("wetext").setLevel(logging.WARNING)
            # Grammars compiled without remove_interjections/remove_erhua. They are cached in
            # ~/.cache, not inside the installed package (which may be read-only).
            cache_dir = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
                                     "indextts25", "wetext_tagger")
            os.makedirs(cache_dir, exist_ok=True)
            self.zh_normalizer = NormalizerZh(
                cache_dir=cache_dir, remove_interjections=False, remove_erhua=False, overwrite_cache=False
            )
            self.en_normalizer = NormalizerEn(overwrite_cache=False)

    G2P_PRONUNCIATION_ANNOTATION_PATTERN = re.compile(r'<([^|>\n]+)\|([^>\n]+)>')

    def _protect_pronunciation_annotations(self, text: str):
        """
        Called before normalize: replaces <word|pronunciation> annotations with letter-only
        placeholders so the normalizer doesn't expand the digits/symbols inside them (e.g. XING2 -> XING二).
        Returns (replaced text, placeholder dict).
        """
        placeholders = {}
        def _idx_to_alpha(n):
            s = ''
            while True:
                s = chr(ord('a') + n % 26) + s
                n = n // 26 - 1
                if n < 0:
                    break
            return s
        def _replacer(m):
            tag = _idx_to_alpha(len(placeholders))
            key = f'PRONPLACEHOLDER{tag}PRONPLACEHOLDER'
            placeholders[key] = m.group(0)
            return key
        text = self.G2P_PRONUNCIATION_ANNOTATION_PATTERN.sub(_replacer, text)
        return text, placeholders

    @staticmethod
    def _restore_pronunciation_annotations(text: str, placeholders: dict) -> str:
        """Called after normalize: restores the placeholders to the original <word|pronunciation> annotations."""
        for key, val in placeholders.items():
            text = text.replace(key, val)
        return text

    def normalize(self, text: str, lang: str = None) -> str:
        """Normalizes with WeTextProcessing. With lang='en' it does not guess the language:
        use_chinese() sent English text with no letters ('2024') to the Chinese normalizer."""
        if not self.load():
            # Without WeTextProcessing the text passes through as is: returning "" made it
            # synthesize ~2 s of noise without any error.
            return text
        # Protect the G2P pronunciation annotations <word|pronunciation> from the normalizer
        text, _pron_placeholders = self._protect_pronunciation_annotations(text)
        if lang != "en" and self.use_chinese(text):
            text = re.sub(TextNormalizer.ENGLISH_CONTRACTION_PATTERN, r"\1 is", text, flags=re.IGNORECASE)
            # Apply the term glossary (highest priority, before any protection)
            if self.enable_glossary:
                text = self.apply_glossary_terms(text, lang="zh")
            # Protect technical terms (e.g. GPT-5-nano) from being mishandled by the Chinese normalizer
            replaced_text, tech_list = self.save_tech_terms(text.rstrip())
            replaced_text, pinyin_list = self.save_pinyin_tones(replaced_text)

            replaced_text, original_name_list = self.save_names(replaced_text)
            try:
                result = self.zh_normalizer.normalize(replaced_text)
            except Exception:
                result = replaced_text  # unnormalized, never empty
                logger.warning("WeTextProcessing (zh) failed, using the raw text", exc_info=True)
            # Restore person names
            result = self.restore_names(result, original_name_list)
            # Restore pinyin tones
            result = self.restore_pinyin_tones(result, pinyin_list)
            # Restore technical terms
            result = self.restore_tech_terms(result, tech_list)
            pattern = re.compile("|".join(re.escape(p) for p in self.zh_char_rep_map.keys()))
            result = pattern.sub(lambda x: self.zh_char_rep_map[x.group()], result)
        else:
            try:
                text = re.sub(TextNormalizer.ENGLISH_CONTRACTION_PATTERN, r"\1 is", text, flags=re.IGNORECASE)
                # Apply the term glossary (highest priority, before any protection)
                if self.enable_glossary:
                    text = self.apply_glossary_terms(text, lang="en")
                # Protect technical terms (e.g. GPT-5-Nano) from being mishandled by the English normalizer
                replaced_text, tech_list = self.save_tech_terms(text)
                result = self.en_normalizer.normalize(replaced_text)
                # Restore technical terms
                result = self.restore_tech_terms(result, tech_list)
            except Exception:
                result = text
                logger.warning("WeTextProcessing (en) failed, using the raw text", exc_info=True)
            pattern = re.compile("|".join(re.escape(p) for p in self.char_rep_map.keys()))
            result = pattern.sub(lambda x: self.char_rep_map[x.group()], result)

        # Restore the G2P pronunciation annotations
        result = self._restore_pronunciation_annotations(result, _pron_placeholders)
        return result

    def correct_pinyin(self, pinyin: str):
        """
        Rewrites the u/ü final of pinyin syllables with a j/q/x initial as v
        e.g. ju -> jv , que -> qve, xün -> xvn
        """
        if pinyin[0] not in "jqxJQX":
            return pinyin
        # Match j/q/x syllables whose final is u/ü
        pattern = r"([jqx])[uü](n|e|an)*(\d)"
        repl = r"\g<1>v\g<2>\g<3>"
        pinyin = re.sub(pattern, repl, pinyin, flags=re.IGNORECASE)
        return pinyin.upper()

    def save_names(self, original_text):
        """
        Replaces person names with the placeholders <n_a>, <n_b>, ...
        e.g. 克里斯托弗·诺兰 -> <n_a>
        """
        # Person names
        name_pattern = re.compile(TextNormalizer.NAME_PATTERN, re.IGNORECASE)
        original_name_list = re.findall(name_pattern, original_text)
        if len(original_name_list) == 0:
            return (original_text, None)
        original_name_list = list(set("".join(n) for n in original_name_list))
        transformed_text = original_text
        # Replace each name with a placeholder <n_a>, <n_b>, ...
        for i, name in enumerate(original_name_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(name, f"<n_{number}>")

        return transformed_text, original_name_list

    def restore_names(self, normalized_text, original_name_list):
        """
        Restores the placeholders to the original names
        e.g. <n_a> -> original_name_list[0]
        """
        if not original_name_list or len(original_name_list) == 0:
            return normalized_text

        transformed_text = normalized_text
        # Replace the placeholders <n_a>, <n_b>, ... with the names
        for i, name in enumerate(original_name_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(f"<n_{number}>", name)
        return transformed_text

    def save_tech_terms(self, original_text):
        """
        Protects the hyphens in technical terms so the Chinese normalizer doesn't read them as minus signs
        Strategy: replace the hyphens in the term with the special placeholder <H>; digits are still normalized
        e.g. GPT-5-nano -> GPT<H>5<H>nano, then 5 is converted to 五 (five)
        Finally restored as: GPT-五-nano
        """
        tech_pattern = re.compile(TextNormalizer.TECH_TERM_PATTERN)
        original_tech_list = tech_pattern.findall(original_text)
        if len(original_tech_list) == 0:
            return (original_text, None)

        # Deduplicate and sort by length, longest first (replacing short matches first causes problems)
        original_tech_list = sorted(set(original_tech_list), key=len, reverse=True)
        transformed_text = original_text

        # Replace the hyphens in each term with the placeholder <H>
        for term in original_tech_list:
            # Replace GPT-5-nano with GPT<H>5<H>nano
            protected_term = term.replace("-", "<H>")
            transformed_text = transformed_text.replace(term, protected_term)

        return transformed_text, original_tech_list

    def restore_tech_terms(self, normalized_text, original_tech_list):
        """
        Restores the hyphens in technical terms
        Turns the placeholder <H> back into a hyphen -
        and also removes any extra spaces the normalizer may have added around the placeholder
        """
        if not original_tech_list or len(original_tech_list) == 0:
            return normalized_text

        # Strip any whitespace around <H>, then turn it back into a hyphen
        # Handled patterns: " <H> " -> "-", " <H>" -> "-", "<H> " -> "-", "<H>" -> "-"
        transformed_text = re.sub(r'\s*<H>\s*', '-', normalized_text)
        return transformed_text

    def apply_glossary_terms(self, text, lang="zh"):
        """
        Applies the term glossary, replacing technical terms with their reading in the given language

        Args:
            text: text to process
            lang: language, "zh" or "en"

        Returns:
            the processed text

        Example:
            "M.2 NVMe SSD" -> (zh) "M 二 NVMe SSD"
            "M.2 NVMe SSD" -> (en) "M dot two NVMe SSD"
        """
        if not self.term_glossary:
            return text

        # Sort terms by length, longest first, so a short term matching first can't block a longer one
        # e.g. "PCIe 5.0" must be matched before "PCIe"
        sorted_terms = sorted(self.term_glossary.keys(), key=len, reverse=True)
        @lru_cache(maxsize=42)
        def get_term_pattern(term: str):
            return re.compile(re.escape(term), re.IGNORECASE)
        transformed_text = text
        for term in sorted_terms:
            term_value = self.term_glossary[term]
            if isinstance(term_value, dict):
                replacement = term_value.get(lang, term_value.get(lang, term))
            else:
                replacement = term_value
            # Case-insensitive regex replacement
            pattern = get_term_pattern(term)
            transformed_text = pattern.sub(replacement, transformed_text)

        return transformed_text

    def load_glossary_from_yaml(self, glossary_path):
        """
        Loads the term glossary from a YAML file

        Args:
            glossary_path: path to the YAML file

        Example:
            normalizer.load_glossary_from_yaml("checkpoints/glossary.yaml")

        YAML file format:
            M.2:
              en: M dot two
              zh: M 二
            NVMe: N-V-M-E  # same reading in Chinese and English
        """
        if glossary_path and os.path.exists(glossary_path):
            import yaml
            with open(glossary_path, encoding='utf-8') as f:
                external_glossary = yaml.safe_load(f)
                if external_glossary and isinstance(external_glossary, dict):
                    self.term_glossary = external_glossary
                    return True
        return False

    def save_pinyin_tones(self, original_text):
        """
        Replaces tone-numbered pinyin (e.g. xuan4) with the placeholders <pinyin_a>, <pinyin_b>, ...
        e.g. xuan4 -> <pinyin_a>
        """
        # Initial + final + tone digit
        origin_pinyin_pattern = re.compile(TextNormalizer.PINYIN_TONE_PATTERN, re.IGNORECASE)
        original_pinyin_list = re.findall(origin_pinyin_pattern, original_text)
        if len(original_pinyin_list) == 0:
            return (original_text, None)
        original_pinyin_list = list(set("".join(p) for p in original_pinyin_list))
        transformed_text = original_text
        # Replace with the placeholders <pinyin_a>, <pinyin_b>, ...
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(pinyin, f"<pinyin_{number}>")

        # print("original_text: ", original_text)
        # print("transformed_text: ", transformed_text)
        return transformed_text, original_pinyin_list

    def restore_pinyin_tones(self, normalized_text, original_pinyin_list):
        """
        Restores the placeholders to the original pinyin with its tone digit (1-5)
        e.g. <pinyin_a> -> original_pinyin_list[0]
        """
        if not original_pinyin_list or len(original_pinyin_list) == 0:
            return normalized_text

        transformed_text = normalized_text
        # Replace the placeholders <pinyin_a>, <pinyin_b>, ...
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            pinyin = self.correct_pinyin(pinyin)
            transformed_text = transformed_text.replace(f"<pinyin_{number}>", pinyin)
        # print("normalized_text: ", normalized_text)
        # print("transformed_text: ", transformed_text)
        return transformed_text

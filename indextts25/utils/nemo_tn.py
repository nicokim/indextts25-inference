#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TTS front-end text normalization (TN).

Uses ``nemo_text_processing`` to expand numbers/symbols/dates/currency and other
non-standard words into speakable text (e.g. ``"25%"`` -> ``"twenty five percent"``).

Design notes:
- **Input is an upstream service language code** (ISO 639-1 style short code, e.g.
  ar/zh/es/en/ja); ``_SERVICE_TO_NEMO`` maps it to NeMo's language code. ISO 639-3
  codes sent directly by upstream (arb/arz/...) are first folded back to the service code.
- **Not every language is normalized** (e.g. ja: NeMo >=1.2 has a grammar, but it is
  deliberately not enabled; see ``_SERVICE_TO_NEMO``). Such text passes through unchanged.
- **Lazy load + cache**: ``Normalizer`` grammars are slow to build (seconds); cached per language.
- **Degrade on failure**: if ``nemo_text_processing`` is missing, the grammar build fails,
  or normalize raises, a warning is logged and the raw text is returned; never aborts.
"""

import logging
import os
import pickle
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Compiling a language's grammar takes ~25 s on every startup; the already-built
# Normalizer is pickled and loads in ~0.6 s. NeMo's cache_dir is not used: exporting
# re-optimizes the graph and some decimals ending in 0 are read differently
# ("1,80" -> "uno coma ocho cero"); the pickle is exactly the in-memory object.
_CACHE_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "indextts25", "nemo_tn")


def _build_normalizer(nemo_lang: str, input_case: str):
    from importlib.metadata import version

    from nemo_text_processing.text_normalization.normalize import Normalizer

    path = os.path.join(_CACHE_DIR, f"{nemo_lang}-{input_case}-nemo{version('nemo_text_processing')}"
                                    f"-pynini{version('pynini')}.pkl")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:  # no cache, or corrupt/from another version: rebuild it
        pass
    normalizer = Normalizer(input_case=input_case, lang=nemo_lang)
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(normalizer, f)
        os.replace(tmp, path)  # atomic: two processes starting together don't clobber each other
    except OSError as e:  # no write permission: keep working, without the cache
        logger.warning(f"could not cache the NeMo grammar in {_CACHE_DIR}: {e}")
    return normalizer


# ---------------------------------------------------------------------------
# Language code mapping: upstream service code / ISO 639-3 -> NeMo TN language code
#
# Only languages with a TN grammar we enable are listed; others (e.g. ja) skip TN.
# NeMo language codes: see nemo_text_processing.text_normalization.normalize.Normalizer(lang=...).
# Before adding a language, verify the installed nemo version actually has its TN grammar.
# ---------------------------------------------------------------------------

# Service code (ISO 639-1 style) -> NeMo language code
_SERVICE_TO_NEMO: Dict[str, str] = {
    "ar": "ar",
    "zh": "zh",
    "es": "es",
    "en": "en",
    # "ja" is left out on purpose: NeMo >=1.2 ships a Japanese grammar, but enabling it
    # changes the output and it was not validated by ear. Japanese passes unnormalized.
}

# ISO 639-3 -> service code
_ISO3_TO_SERVICE: Dict[str, str] = {
    "arb": "ar",  # standard arabic
    "arz": "ar",  # egyptian arabic
    "ary": "ar",  # moroccan arabic
    "ars": "ar",  # najdi arabic
    "zho": "zh",
    "cmn": "zh",
    "spa": "es",
    "eng": "en",
    "jpn": "ja",
}


def _to_nemo_lang(lang: Optional[str]) -> Optional[str]:
    """Map an upstream language code to its NeMo TN code; None if TN is unsupported."""
    if not lang:
        return None
    key = lang.lower()
    if key in _SERVICE_TO_NEMO:
        return _SERVICE_TO_NEMO[key]
    # Upstream may pass ISO 639-3 directly (e.g. arb / spa): fold back to the service code
    svc = _ISO3_TO_SERVICE.get(key)
    if svc and svc in _SERVICE_TO_NEMO:
        return _SERVICE_TO_NEMO[svc]
    return None


class TextNormalizer:
    """Wrapper that lazily builds and caches one NeMo ``Normalizer`` per language.

    Used as a singleton (``get_text_normalizer()`` below): grammars are built once and reused.

    Args:
        input_case: NeMo casing mode: ``"cased"`` keeps case (default; good for proper
            nouns / mixed-language text), ``"lower_cased"`` lowercases before normalizing.
    """

    def __init__(self, input_case: str = "cased"):
        self.input_case = input_case
        # nemo_lang -> Normalizer; None = language unavailable (already tried and failed)
        self._cache: Dict[str, Optional[object]] = {}

    def _get_normalizer(self, nemo_lang: str):
        """Return the cached Normalizer; build it once, caching None on failure to avoid retries."""
        if nemo_lang in self._cache:
            return self._cache[nemo_lang]

        normalizer = None
        try:
            # NeMo leaves its logger at INFO with its own handler ("so applications can
            # adjust it"): without this, every grammar load prints to stderr.
            logging.getLogger("NeMo-text-processing").setLevel(logging.WARNING)
            normalizer = _build_normalizer(nemo_lang, self.input_case)
            logger.info(f"nemo Normalizer(lang={nemo_lang}) initialized")
        except Exception as e:
            logger.warning(
                f"build nemo Normalizer(lang={nemo_lang}) failed -> "
                f"skip text normalization for this language: {e}"
            )
            normalizer = None

        self._cache[nemo_lang] = normalizer
        return normalizer

    def normalize(self, text: Optional[str], lang: Optional[str]) -> Optional[str]:
        """Normalize ``text``.

        Unsupported language / NeMo unavailable / normalize raises: returns ``text`` as is.

        Args:
            text: text to normalize.
            lang: upstream language code (service code or ISO 639-3).

        Returns:
            The normalized text, or the original text if it cannot be processed.
        """
        if not text:
            return text

        nemo_lang = _to_nemo_lang(lang)
        if nemo_lang is None:
            # Language-agnostic mode or no NeMo TN for this language (e.g. ja): skip
            return text

        normalizer = self._get_normalizer(nemo_lang)
        if normalizer is None:
            return text

        try:
            return normalizer.normalize(text, verbose=False)
        except Exception as e:
            logger.warning(
                f"text normalization failed (lang={lang}->{nemo_lang}) -> "
                f"use raw text: {e}"
            )
            return text


_DEFAULT_NORMALIZER: Optional[TextNormalizer] = None


def get_text_normalizer(input_case: str = "cased") -> TextNormalizer:
    """Return the process-wide shared ``TextNormalizer`` singleton."""
    global _DEFAULT_NORMALIZER
    if _DEFAULT_NORMALIZER is None:
        _DEFAULT_NORMALIZER = TextNormalizer(input_case=input_case)
    return _DEFAULT_NORMALIZER


def normalize_text(text: Optional[str], lang: Optional[str]) -> Optional[str]:
    """Convenience entry point: normalize ``text`` for ``lang`` with the shared singleton."""
    return get_text_normalizer().normalize(text, lang)

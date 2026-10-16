"""IndexTTS-2.5 inference engine (derived from index-tts, commit ee40fa7).

`IndexTTS2` loads the models and runs the full pipeline:

1. Reference-voice conditioning, cached per file: w2v-bert (layer 17)
   + CAMPPlus (speaker style) + reference mel.
2. Text: normalization (NeMo; WeTextProcessing for zh/en if available), upper- or
   lowercasing by language, pronunciation annotations and token-based splitting.
3. Per segment: GPT (semantic codes) -> codec -> length regulator -> CFM/DiT
   (mel) -> BigVGAN (waveform at 22.05 kHz).

The public API is `indextts25.api.IndexTTS25`.
"""
import os
import random
import re
import time
import warnings

import librosa
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors import safe_open
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertConfig, Wav2Vec2BertModel

from indextts25.codec.models import EnhancedCodec
from indextts25.gpt.model_v2 import UnifiedVoice
from indextts25.s2mel.modules.audio import mel_spectrogram
from indextts25.s2mel.modules.bigvgan.bigvgan import BigVGAN
from indextts25.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts25.s2mel.modules.commons import MyModel
from indextts25.utils.checkpoint import fold_weight_norm, load_checkpoint
from indextts25.utils.common import kaldi_fbank, resample, save_pcm_wav
from indextts25.utils.front import TextNormalizer
from indextts25.utils.ja_g2p import JapaneseG2PProcessor
from indextts25.utils.model_download import ensure_models_available
from indextts25.utils.nemo_tn import normalize_text as nemo_text_normalize
from indextts25.utils.tokenizer import LANGUAGE_DICT, get_encoding, lang_to_token

SAMPLE_RATE = 22050
# Distinct voices/emotions kept in the cache (~6 MB of GPU memory each).
_CACHE_MAX = 8  # ponytail: FIFO; switch to LRU if voices rotate a lot

PRONUNCIATION_ANNOTATION_PATTERN = re.compile(r'<([^|>\n]+)\|([^>\n]+)>')
"""
匹配发音标注格式：<文字|发音>
例如：<going|G OW1 . IH0 NG>，<行|XING2>
"""


def is_kana(s: str) -> bool:
    hira = re.compile(r'^[぀-ゟ]+$')
    kata = re.compile(r'^[゠-ヿ]+$')
    if hira.fullmatch(s):
        return True
    elif kata.fullmatch(s):
        return True
    return False


def apply_pronunciation_annotations(text: str) -> str:
    """
    处理发音标注格式 <文字|发音>，转换为特殊token包裹的发音
    - 文字含中文 -> <|SPECIAL_TOKEN_2|>发音<|SPECIAL_TOKEN_2|>
    - 文字为英文 -> <|SPECIAL_TOKEN_1|>发音<|SPECIAL_TOKEN_1|>
    发音内容统一转大写

    例如：
        <going|G OW1 . IH0 NG> -> <|SPECIAL_TOKEN_1|>G OW1 . IH0 NG<|SPECIAL_TOKEN_1|>
        <行|XING2>              -> <|SPECIAL_TOKEN_2|>XING2<|SPECIAL_TOKEN_2|>
    """
    def _replace(match):
        word = match.group(1)
        pronunciation = match.group(2).upper()
        has_chinese = bool(re.search(r"[一-鿿]", word))
        if is_kana(pronunciation):
            return f' {pronunciation} '
        else:
            token = 'SPECIAL_TOKEN_2' if has_chinese else 'SPECIAL_TOKEN_1'
            return f'<|{token}|>{pronunciation}<|{token}|>'
    return PRONUNCIATION_ANNOTATION_PATTERN.sub(_replace, text)


def find_most_similar_cosine(query_vector, matrix):
    similarities = F.cosine_similarity(query_vector.float(), matrix.float(), dim=1)
    return torch.argmax(similarities)


def _audio_key(path):
    # Absolute path + mtime: overwriting the reference file invalidates the cache.
    path = os.path.abspath(path)
    return path, os.stat(path).st_mtime_ns


def _cache_put(cache, key, value):
    cache[key] = value
    while len(cache) > _CACHE_MAX:
        cache.pop(next(iter(cache)))
    return value


class IndexTTS2:
    def __init__(self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_bf16=False,
                 device=None, use_accel=False):
        """
        Args:
            cfg_path: model YAML. `gpt_checkpoint:` lets it point to a finetuned GPT.
            model_dir: folder with gpt.pth, s2mel.pth, codec.pth, the .tiktoken file and hf_cache/.
            use_bf16: GPT in bf16 (the accel engine shares those weights; flash-attn runs in
                fp16). Ignored on CPU and MPS.
            device: 'cuda:0', 'cpu', ... Defaults to CUDA, XPU, MPS or CPU, in that order.
            use_accel: accelerated GPT2 engine (flash-attn, paged KV cache, CUDA graphs).
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                device = "xpu"
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.use_bf16 = use_bf16 and not device.startswith(("cpu", "mps"))
        self.dtype = torch.bfloat16 if self.use_bf16 else None

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        # The modules' random init (which the checkpoints overwrite) consumes the RNG;
        # fork_rng restores it, so seeding before construction stays reproducible.
        cuda_idx = [torch.device(device).index or 0] if device.startswith("cuda") else []
        with torch.random.fork_rng(devices=cuda_idx):
            self._load_models(use_accel)

        # Conditioning cached by (absolute path, mtime) of the reference audio.
        self._spk_cache = {}  # -> (spk_cond_emb, style, prompt_condition, ref_mel)
        self._emo_cache = {}  # -> emo_cond_emb
        self._emovec_cache = {}  # (voice, emotion, alpha) -> GPT emotion vector

    def _load_models(self, use_accel):
        # Models are built directly on the device: the random initialization (which
        # the checkpoints overwrite) used to cost ~6 s on CPU. The later .to() only
        # moves what some legacy constructor (torch.Tensor(...)) created on CPU.
        with torch.device(self.device):
            self.gpt = UnifiedVoice(**self.cfg.gpt, use_accel=use_accel)
        load_checkpoint(self.gpt, os.path.join(self.model_dir, self.cfg.gpt_checkpoint))
        self.gpt.to(self.device).eval()
        if self.use_bf16:
            self.gpt.bfloat16()
        self.gpt.post_init_gpt2_config()

        aux = ensure_models_available(self.model_dir)
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(aux["w2v_bert"], local_files_only=True)
        # get_emb only reads the output of layer 17: the other 7 took 0.65 GB of VRAM
        # and ~30 % of the compute, with no effect on the result. It is built on the device
        # and only the weights it uses are read (from_pretrained, on transformers 5.x, also
        # drew a progress bar on stderr). Loading is strict.
        w2v_config = Wav2Vec2BertConfig.from_pretrained(aux["w2v_bert"], num_hidden_layers=17)
        with torch.device(self.device):
            self.semantic_model = Wav2Vec2BertModel(w2v_config)
        wanted = self.semantic_model.state_dict().keys()
        with safe_open(os.path.join(aux["w2v_bert"], "model.safetensors"), "pt") as f:
            self.semantic_model.load_state_dict({k: f.get_tensor(k) for k in f.keys() if k in wanted})
        self.semantic_model.eval()
        stat_mean_var = torch.load(os.path.join(self.model_dir, self.cfg.w2v_stat))
        self.semantic_mean = stat_mean_var["mean"].to(self.device)
        self.semantic_std = torch.sqrt(stat_mean_var["var"]).to(self.device)

        with torch.device(self.device):
            self.semantic_codec = EnhancedCodec(**self.cfg.semantic_codec)
        self.semantic_codec.load_checkpoint(os.path.join(self.model_dir, "codec.pth"), self.device)
        self.semantic_codec.to(self.device).eval()

        with torch.device(self.device):
            self.s2mel = MyModel(self.cfg.s2mel)
        # Strict loading: the upstream loader filtered by shape with strict=False, so an
        # incompatible checkpoint silently ran with randomly initialized layers.
        net = torch.load(os.path.join(self.model_dir, self.cfg.s2mel_checkpoint), map_location="cpu", mmap=True)["net"]
        for name, module in self.s2mel.models.items():
            module.load_state_dict(fold_weight_norm(net[name], self.device))
        self.s2mel.to(self.device)
        self.s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        self.s2mel.eval()

        with torch.device(self.device):
            self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus_model.load_state_dict(torch.load(aux["campplus"], map_location="cpu"))
        self.campplus_model.to(self.device).eval()

        self.bigvgan = BigVGAN.from_dir(aux["bigvgan"], self.device).eval()

        self.tokenizer = get_encoding("multilingual_zh_ja_yue_char_del", model_dir=self.model_dir)
        # zh/en (WeTextProcessing) and the Japanese G2P (fugashi) load lazily on first use.
        self._ja_text_process = None
        self.text_process = TextNormalizer(enable_glossary=True)
        glossary_path = os.path.join(self.model_dir, "glossary.yaml")
        if os.path.exists(glossary_path):
            self.text_process.load_glossary_from_yaml(glossary_path)

        self.emo_num = list(self.cfg.emo_num)
        emo_matrix = torch.load(os.path.join(self.model_dir, self.cfg.emo_matrix)).to(self.device)
        spk_matrix = torch.load(os.path.join(self.model_dir, self.cfg.spk_matrix)).to(self.device)
        self.emo_matrix = torch.split(emo_matrix, self.emo_num)
        self.spk_matrix = torch.split(spk_matrix, self.emo_num)

        spect = self.cfg.s2mel['preprocess_params']['spect_params']
        mel_fn_args = {
            "n_fft": spect['n_fft'],
            "win_size": spect['win_length'],
            "hop_size": spect['hop_length'],
            "num_mels": spect['n_mels'],
            "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
            "fmin": spect.get('fmin', 0),
            "fmax": None if spect.get('fmax', "None") == "None" else 8000,
            "center": False
        }
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

    @property
    def ja_text_process(self):
        if self._ja_text_process is None:
            self._ja_text_process = JapaneseG2PProcessor(g2p_ratio=0)
        return self._ja_text_process

    @torch.no_grad()
    def get_emb(self, input_features, attention_mask):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat

    def _load_and_cut_audio(self, audio_path, max_audio_length_seconds, verbose=False, sr=None):
        if not sr:
            audio, sr = librosa.load(audio_path)  # librosa resamples to 22050 Hz
        else:
            audio, _ = librosa.load(audio_path, sr=sr)
        audio = torch.tensor(audio).unsqueeze(0)
        max_audio_samples = int(max_audio_length_seconds * sr)

        if audio.shape[1] > max_audio_samples:
            if verbose:
                print(f"Audio too long ({audio.shape[1]} samples), truncating to {max_audio_samples} samples")
            audio = audio[:, :max_audio_samples]
        return audio, sr

    def _speaker_conditioning(self, path, key, verbose=False):
        if key in self._spk_cache:
            return self._spk_cache[key]
        audio, sr = self._load_and_cut_audio(path, 15, verbose)
        audio_16k = resample(audio, sr, 16000)
        inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        spk_cond_emb = self.get_emb(inputs["input_features"].to(self.device),
                                    inputs["attention_mask"].to(self.device))
        ref_mel = self.mel_fn(audio.to(spk_cond_emb.device).float())
        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
        feat = kaldi_fbank(audio_16k.to(ref_mel.device), num_mel_bins=80, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        style = self.campplus_model(feat.unsqueeze(0))  # global speaker style [1, 192]
        prompt_condition = self.s2mel.models['length_regulator'](spk_cond_emb, ylens=ref_target_lengths)
        return _cache_put(self._spk_cache, key, (spk_cond_emb, style, prompt_condition, ref_mel))

    def _emotion_conditioning(self, path, key, verbose=False):
        if key in self._emo_cache:
            return self._emo_cache[key]
        emo_audio, _ = self._load_and_cut_audio(path, 15, verbose, sr=16000)
        inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
        emo_cond_emb = self.get_emb(inputs["input_features"].to(self.device),
                                    inputs["attention_mask"].to(self.device))
        return _cache_put(self._emo_cache, key, emo_cond_emb)

    def _prepare_text(self, text, lang, text_normalization=True, verbose=False):
        lang = lang.lower()
        sub = lambda t, keep='': self.text_process.clean_pattern.sub(
            lambda x: x.group() if x.group() in keep else self.text_process.char_rep_map[x.group()], t)
        # char_rep_map turns ":" into ",". Applied BEFORE normalizing, "14:30" was read as
        # "catorce coma treinta" (FIX over upstream). For es/ja it runs after; for zh/en the
        # other symbols ('~', '—', line breaks, quotes) do have to go before the
        # normalizer, so only ':' is left for after.
        if lang not in ('es', 'ja'):
            text = sub(text, ':：' if text_normalization and lang in ('zh', 'zhen', 'en') else '')
        if text_normalization:
            if lang in ('zh', 'zhen', 'en') and self.text_process.load():  # WeTextProcessing, extra [zh]
                text = self.text_process.normalize(text, lang)
            elif lang in ('es', 'ja', 'en', 'zh'):
                text = nemo_text_normalize(text, lang)
            if verbose:
                print(f'text after normalization: {text}')
        text = sub(text)

        if lang in ('ja', 'zh', 'zhen', 'en'):
            text = text.lower()
        elif lang == 'es':
            text = text.upper()
        text = apply_pronunciation_annotations(text)
        if lang == 'ja':
            text = self.ja_text_process.process_ja_text(text)
        return re.sub(r'<\|([^|]+)\|>', lambda m: f'<|{m.group(1).upper()}|>', text)

    SPLIT_PROTECTED_PATTERN = re.compile(r'<\|SPECIAL_TOKEN_\d+\|>.*?<\|SPECIAL_TOKEN_\d+\|>')

    def _token_len(self, text):
        return len(self.tokenizer.encode(text, allowed_special='all'))

    def _split_atomic_pieces(self, text):
        pieces, pos = [], 0
        for m in self.SPLIT_PROTECTED_PATTERN.finditer(text):
            if m.start() > pos:
                pieces.append((text[pos:m.start()], False))
            pieces.append((m.group(0), True))
            pos = m.end()
        if pos < len(text):
            pieces.append((text[pos:], False))
        return pieces

    def split_text_by_tokens(self, text, max_tokens, lang_prefix=""):
        """Split the text into segments of up to `max_tokens` (language prefix included),
        cutting at punctuation and never breaking <|SPECIAL_TOKEN_n|> annotations."""
        capacity = self.gpt.text_pos_embedding.emb.num_embeddings
        budget = min(max_tokens, capacity - 2) - self._token_len(lang_prefix)
        budget = max(1, budget)
        if self._token_len(text) <= budget:
            return [text]

        chunks = []
        for piece, atomic in self._split_atomic_pieces(text):
            if atomic:
                chunks.append(piece)
                continue
            for part in re.split(r'(?<=[，。！？、；：,\.!\?;:\n])', piece):
                if not part:
                    continue
                if self._token_len(part) <= budget:
                    chunks.append(part)
                    continue
                # A stretch without punctuation that doesn't fit: cut between words. Only a
                # word that doesn't fit by itself (e.g. zh without spaces) is cut per character.
                current = ""
                for word in re.findall(r'\S+\s*|\s+', part):
                    for ch in ([word] if self._token_len(word) <= budget else word):
                        if current and self._token_len(current + ch) > budget:
                            chunks.append(current)
                            current = ch
                        else:
                            current += ch
                if current:
                    chunks.append(current)

        segments, current = [], ""
        for chunk in chunks:
            if current and self._token_len(current + chunk) > budget:
                segments.append(current)
                current = chunk
            else:
                current += chunk
        if current:
            segments.append(current)
        return segments or [text]

    def infer(self, spk_audio_prompt, text, output_path, lang, **kwargs):
        """Synthesize the full text. With `output_path`, write a 16-bit PCM WAV and return
        the path; otherwise return (sample_rate, int16 ndarray [N, 1])."""
        for result in self.infer_generator(spk_audio_prompt, text, output_path, lang, **kwargs):
            return result

    @torch.no_grad()
    def infer_generator(self, spk_audio_prompt, text, output_path, lang,
                        emo_audio_prompt=None, emo_alpha=1.0, emo_vector=None, use_random=False,
                        interval_silence=200, verbose=False, max_text_tokens_per_segment=120,
                        stream_return=False, duration_factor=1.0, text_normalization=True,
                        emit_text=False, **generation_kwargs):
        """Full pipeline. With `stream_return=True`, yield each segment as soon as it is
        ready (and `interval_silence` ms of silence between segments); otherwise one result.

        `emit_text=True` yields (audio, text) instead of audio only. The text is the
        model input: normalized, with numbers spelled out, and uppercased for Spanish.
        """
        start_time = time.perf_counter()
        if lang.lower() not in LANGUAGE_DICT and lang.lower() != "zhen":
            raise ValueError(f"lang={lang!r}: use an ISO 639-1 code such as 'es', 'en', 'zh' or 'ja'")
        if verbose:
            print(f"origin text:{text}, spk_audio_prompt:{spk_audio_prompt}, "
                  f"emo_audio_prompt:{emo_audio_prompt}, emo_alpha:{emo_alpha}, emo_vector:{emo_vector}")

        if emo_vector is not None:
            # With an emotion vector there is no emotion audio. Since the vector skips
            # merge_emovec's blending, alpha is applied to it here.
            emo_audio_prompt = None
            emo_vector_scale = max(0.0, min(1.0, emo_alpha))
            if emo_vector_scale != 1.0:
                emo_vector = [int(x * emo_vector_scale * 10000) / 10000 for x in emo_vector]
        if emo_audio_prompt is None:
            # Without emotion audio, the emotion comes from the voice audio, with alpha 1.
            emo_audio_prompt = spk_audio_prompt
            emo_alpha = 1.0

        spk_key, emo_key = _audio_key(spk_audio_prompt), _audio_key(emo_audio_prompt)
        spk_cond_emb, style, prompt_condition, ref_mel = self._speaker_conditioning(spk_audio_prompt, spk_key, verbose)
        emo_cond_emb = self._emotion_conditioning(emo_audio_prompt, emo_key, verbose)

        if emo_vector is not None:
            weight_vector = torch.tensor(emo_vector, device=self.device)
            if use_random:
                random_index = [random.randint(0, x - 1) for x in self.emo_num]
            else:
                random_index = [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]
            emo_matrix = torch.cat([tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)], 0)
            emovec_mat = torch.sum(weight_vector.unsqueeze(1) * emo_matrix, 0).unsqueeze(0)

        text = self._prepare_text(text, lang, text_normalization, verbose)
        lang_prefix = f'<|{lang.lower()}|> '
        segments = self.split_text_by_tokens(text, max_text_tokens_per_segment, lang_prefix)
        segment_tokens = []
        for seg_text in segments:
            toks = self.tokenizer.encode(lang_prefix + seg_text, allowed_special='all')
            # No stop token at the end: prepare_gpt_inputs adds it. An extra one here became a
            # left-padding token, which the flash engine also attended to.
            segment_tokens.append(torch.IntTensor(toks).unsqueeze(0).to(self.device))
        lang_id = torch.LongTensor([lang_to_token(lang)]).to(self.device)
        if verbose:
            print("segments count:", len(segments))
            print("max_text_tokens_per_segment:", max_text_tokens_per_segment)
            print(*segments, sep="\n")

        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 0.8)
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)
        # Diffusion Euler steps and CFG weight: the s2mel cost scales with the step count.
        diffusion_steps = generation_kwargs.pop("diffusion_steps", 25)
        inference_cfg_rate = generation_kwargs.pop("inference_cfg_rate", 0.7)

        autocast = torch.amp.autocast(torch.device(self.device).type, enabled=self.dtype is not None, dtype=self.dtype)
        # The emotion vector depends only on the voice, the emotion and alpha: it is computed
        # once and reused across segments and across calls. (The lengths are the feature
        # dimension, not the frame count: that is how upstream does it and, with the 15 s
        # cap, the mask is all ones.)
        emovec_key = (spk_key, emo_key, emo_alpha)
        emovec = self._emovec_cache.get(emovec_key)
        if emovec is None:
            with autocast:
                emovec = self.gpt.merge_emovec(
                    spk_cond_emb,
                    emo_cond_emb,
                    torch.tensor([spk_cond_emb.shape[-1]], device=self.device),
                    torch.tensor([emo_cond_emb.shape[-1]], device=self.device),
                    alpha=emo_alpha
                )
            _cache_put(self._emovec_cache, emovec_key, emovec)
        if emo_vector is not None:
            with autocast:
                emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec

        def now():
            # Per-stage timings (verbose) need a GPU sync; otherwise each stage absorbs
            # part of the previous stage's asynchronous work.
            if verbose and torch.cuda.is_available():
                torch.cuda.synchronize()
            return time.perf_counter()

        wavs = []
        gpt_gen_time = s2mel_time = bigvgan_time = 0
        has_warned = False
        silence = torch.zeros(1, int(SAMPLE_RATE * interval_silence / 1000.0))
        for seg_idx, text_tokens in enumerate(segment_tokens):
            if verbose:
                print(f"text_tokens shape: {text_tokens.shape}")
            m_start_time = now()
            with autocast:
                codes = self.gpt.inference_speech(
                    text_tokens,
                    lang_id,
                    emovec,
                    style,
                    max_generate_length=max_mel_tokens,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    length_penalty=length_penalty,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    **generation_kwargs
                )
            gpt_gen_time += now() - m_start_time
            if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                warnings.warn(
                    f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                    f"Input text tokens: {text_tokens.shape[1]}. "
                    f"Consider reducing `max_text_tokens_per_segment`({max_text_tokens_per_segment}) or increasing `max_mel_tokens`.",
                    category=RuntimeWarning
                )
                has_warned = True
            # Cut at the first stop token (a single GPU synchronization).
            stop = (codes[0] == self.stop_mel_token).nonzero()
            if len(stop):
                codes = codes[:, :stop[0, 0].item()]
            if codes.shape[1] == 0:
                continue  # the GPT stopped right away: nothing to vocode
            if verbose:
                print(f"codes shape: {codes.shape}")

            m_start_time = now()
            S_infer = self.semantic_codec.decode(codes)
            target_lengths = torch.LongTensor([int(S_infer.shape[1] * 1.72 * duration_factor)]).to(codes.device)
            cond = self.s2mel.models['length_regulator'](S_infer, ylens=target_lengths)
            cat_condition = torch.cat([prompt_condition, cond], dim=1)
            vc_target = self.s2mel.models['cfm'].inference(cat_condition,
                                                           torch.LongTensor([cat_condition.size(1)]).to(cond.device),
                                                           ref_mel, style, None, diffusion_steps,
                                                           inference_cfg_rate=inference_cfg_rate)
            vc_target = vc_target[:, :, ref_mel.size(-1):]
            s2mel_time += now() - m_start_time

            m_start_time = now()
            wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
            bigvgan_time += now() - m_start_time

            wav = torch.clamp(32767 * wav, -32767.0, 32767.0).cpu()
            wavs.append(wav)
            if stream_return:
                yield (wav, segments[seg_idx]) if emit_text else wav
                if seg_idx < len(segment_tokens) - 1 and interval_silence > 0:
                    yield (silence, "") if emit_text else silence
        end_time = time.perf_counter()
        if stream_return and not output_path:
            return

        parts = []
        for i, w in enumerate(wavs):
            if i and interval_silence > 0:
                parts.append(silence)
            parts.append(w)
        wav = torch.cat(parts, dim=1) if parts else torch.zeros(1, 0)
        if verbose:
            wav_length = wav.shape[-1] / SAMPLE_RATE
            print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
            print(f">> s2mel_time: {s2mel_time:.2f} seconds")
            print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
            print(f">> Total inference time: {end_time - start_time:.2f} seconds")
            print(f">> Generated audio length: {wav_length:.2f} seconds")
            if wav_length:
                print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

        if output_path:
            if os.path.dirname(output_path):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_pcm_wav(output_path, wav, SAMPLE_RATE)
            if verbose:
                print(">> wav file saved to:", output_path)
            if not stream_return:
                yield output_path
        else:
            yield (SAMPLE_RATE, wav.type(torch.int16).numpy().T)

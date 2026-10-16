import math
import wave
from functools import lru_cache

import torch

# Full-scale value used everywhere we convert between normalized waveforms and
# 16-bit PCM. 32767 rather than 32768 so that the positive peak cannot overflow.
PCM16_MAX = 32767.0


def save_pcm_wav(path, wav, sampling_rate):
    """Writes `wav` (PCM scale, values in [-32767, 32767], int or float) as a 16-bit PCM
    WAV. Uses the stdlib: torchaudio.save() routes through TorchCodec since 2.9."""
    wav = wav.detach().to(device="cpu", dtype=torch.float32)
    wav = wav.clamp(-PCM16_MAX, PCM16_MAX)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    channels = wav.shape[0]
    pcm = wav.round().to(torch.int16).transpose(0, 1).contiguous().numpy()
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(int(sampling_rate))
        f.writeframes(pcm.tobytes())


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


# --- Subset of torchaudio (BSD-2-Clause, (c) 2017 Facebook Inc.) ----------------------
# functional._get_sinc_resample_kernel/_apply_sinc_resample_kernel and compliance.kaldi.fbank,
# specialized to the arguments inference uses. Same operations in the same order: the
# result is bit-for-bit identical, without depending on torchaudio (in maintenance mode
# since 2.9; 2.11 was its last release).

@lru_cache(maxsize=8)
def _sinc_kernel(orig: int, new: int, lowpass_filter_width: int = 6, rolloff: float = 0.99):
    base = min(orig, new) * rolloff
    width = math.ceil(lowpass_filter_width * orig / base)
    idx = torch.arange(-width, width + orig, dtype=torch.float64, device="cpu")[None, None] / orig
    t = torch.arange(0, -new, -1, device="cpu")[:, None, None] / new + idx
    t *= base
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)
    window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    t *= math.pi
    kernels = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernels *= window * (base / orig)
    return kernels.to(torch.float32), width


def resample(wav: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
    """== torchaudio.transforms.Resample(orig_freq, new_freq)(wav) (sinc_interp_hann)."""
    if orig_freq == new_freq:
        return wav
    g = math.gcd(int(orig_freq), int(new_freq))
    orig, new = int(orig_freq) // g, int(new_freq) // g
    kernel, width = _sinc_kernel(orig, new)
    shape = wav.size()
    wav = wav.view(-1, shape[-1])
    n, length = wav.shape
    wav = torch.nn.functional.pad(wav, (width, width + orig))
    out = torch.nn.functional.conv1d(wav[:, None], kernel.to(wav.device), stride=orig)
    out = out.transpose(1, 2).reshape(n, -1)
    out = out[..., : torch.ceil(torch.as_tensor(new * length / orig)).long()]
    return out.view(shape[:-1] + out.shape[-1:])


def kaldi_fbank(wav: torch.Tensor, num_mel_bins: int = 80, sample_frequency: float = 16000.0) -> torch.Tensor:
    """== torchaudio.compliance.kaldi.fbank(wav, num_mel_bins=..., dither=0,
    sample_frequency=...) with all other arguments at their defaults (povey window,
    25/10 ms, pre-emphasis 0.97, snip_edges)."""
    device, dtype = wav.device, wav.dtype
    x = wav[0]
    shift = int(sample_frequency * 10.0 * 0.001)
    size = int(sample_frequency * 25.0 * 0.001)
    padded = 1 if size == 0 else 2 ** (size - 1).bit_length()
    m = 1 + (x.size(0) - size) // shift
    frames = x.as_strided((m, size), (shift * x.stride(0), x.stride(0)))
    frames = frames - torch.mean(frames, dim=1).unsqueeze(1)
    prev = torch.nn.functional.pad(frames.unsqueeze(0), (1, 0), mode="replicate").squeeze(0)
    frames = frames - 0.97 * prev[:, :-1]
    frames = frames * torch.hann_window(size, periodic=False, device=device, dtype=dtype).pow(0.85).unsqueeze(0)
    frames = torch.nn.functional.pad(frames.unsqueeze(0), (0, padded - size), mode="constant", value=0).squeeze(0)
    spectrum = torch.fft.rfft(frames).abs().pow(2.0)
    # kaldi.get_mel_banks(num_mel_bins, padded, sample_frequency, 20.0, 0.0) without VTLN
    def mel(f):
        return 1127.0 * math.log(1.0 + f / 700.0)
    lo, hi = mel(20.0), mel(0.5 * sample_frequency)
    delta = (hi - lo) / (num_mel_bins + 1)
    b = torch.arange(num_mel_bins, device="cpu").unsqueeze(1)
    left, center, right = lo + b * delta, lo + (b + 1.0) * delta, lo + (b + 2.0) * delta
    fft_mel = 1127.0 * (1.0 + (sample_frequency / padded) * torch.arange(padded / 2, device="cpu") / 700.0).log()
    fft_mel = fft_mel.unsqueeze(0)
    up, down = (fft_mel - left) / (center - left), (right - fft_mel) / (right - center)
    banks = torch.max(torch.zeros(1, device="cpu"), torch.min(up, down)).to(device=device, dtype=dtype)
    banks = torch.nn.functional.pad(banks, (0, 1), mode="constant", value=0)
    eps = torch.tensor(torch.finfo(torch.float).eps).to(device=device, dtype=dtype)
    return torch.max(torch.mm(spectrum, banks.T), eps).log()

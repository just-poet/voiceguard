"""Bounded decoding and deterministic telephone-channel experiments."""
from io import BytesIO
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

SR = 16000
HOP = 8000


def resample(x, source, target=SR):
    if not 4000 <= source <= 192000:
        raise ValueError('Sample rate must be between 4 and 192 kHz.')
    x = np.asarray(x, dtype=np.float64)
    if source == target:
        return x.copy()
    divisor = gcd(int(source), int(target))
    return resample_poly(x, target // divisor, source // divisor)


def decode(source, max_seconds=10):
    """Decode no more than the limit, checking the header before allocating."""
    stream = BytesIO(source) if isinstance(source, bytes) else Path(source)
    try:
        with sf.SoundFile(stream) as f:
            if not 4000 <= f.samplerate <= 192000 or f.channels > 8:
                raise ValueError('Unsupported sample rate or channel count.')
            if len(f) > max_seconds * f.samplerate:
                raise ValueError(f'Audio must be at most {max_seconds} seconds.')
            source_rate = f.samplerate
            x = f.read(dtype='float32', always_2d=True).mean(axis=1)
    except (sf.LibsndfileError, RuntimeError) as e:
        raise ValueError('Cannot decode this file. Use WAV, FLAC, or OGG.') from e
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Audio is empty or contains invalid samples.')
    return np.clip(resample(x, source_rate), -1, 1), source_rate


def _linear_to_g711(x, codec):
    """G.711 8-bit companding, followed by decoding (actual code words)."""
    pcm = np.clip(np.rint(x * 32768), -32768, 32767).astype(np.int32)
    if codec == 'mulaw':
        sign = np.where(pcm < 0, 0x80, 0)
        magnitude = np.minimum(np.abs(pcm), 32635) + 0x84
        exponent = np.maximum(0, np.floor(np.log2(magnitude)).astype(int) - 7)
        mantissa = (magnitude >> (exponent + 3)) & 0xF
        codes = (~(sign | (exponent << 4) | mantissa)) & 0xFF
        u = (~codes) & 0xFF
        value = (((u & 0xF) << 3) + 0x84) << ((u >> 4) & 7)
        value -= 0x84
        return np.where(u & 0x80, -value, value) / 32768.0
    if codec != 'alaw':
        raise ValueError('Unknown G.711 codec.')
    # G.711 A-law works on 13-bit linear PCM.
    pcm = pcm >> 3
    mask = np.where(pcm >= 0, 0xD5, 0x55)
    magnitude = np.where(pcm >= 0, pcm, -pcm - 1)
    segment = np.searchsorted(np.array([0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF]), magnitude)
    segment = np.minimum(segment, 7)
    mantissa = (magnitude >> np.where(segment < 2, 1, segment)) & 0xF
    codes = ((segment << 4) | mantissa) ^ mask
    a = codes ^ 0x55
    value = (a & 0xF) << 4
    segment = (a & 0x70) >> 4
    value += np.where(segment == 0, 8, 0x108)
    value = value << np.maximum(segment - 1, 0)
    return np.where(a & 0x80, value, -value) / 32768.0


def telephone(x, codec='resample8k'):
    low = resample(x, SR, 8000)
    low = sosfilt(butter(4, [300, 3400], btype='bandpass', fs=8000, output='sos'), low)
    if codec in ('mulaw', 'alaw'):
        low = _linear_to_g711(low, codec)
    elif codec != 'resample8k':
        raise ValueError('Unknown channel.')
    return resample(low, 8000, SR)[:len(x)]


def add_noise(x, snr_db=10, seed=42):
    rng = np.random.default_rng(seed)
    rms = np.sqrt(np.mean(np.square(x)))
    return np.clip(x + rng.normal(0, rms / 10 ** (snr_db / 20), len(x)), -1, 1)


def condition(x, name, seed=42):
    if name == 'clean':
        return x.copy()
    noisy = name.endswith('_noise10')
    channel = name.removesuffix('_noise10')
    y = telephone(x, channel)
    return add_noise(y, 10, seed) if noisy else y


CONDITIONS = ('clean', 'resample8k', 'mulaw', 'alaw', 'resample8k_noise10', 'mulaw_noise10', 'alaw_noise10')

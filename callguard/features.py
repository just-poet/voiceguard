"""A lightweight LFCC baseline; no network or downloaded model required."""
import numpy as np
from scipy.fft import dct
from scipy.signal import stft

from .audio import SR

FEATURE_VERSION = 'lfcc20-delta-spectral-v1'


def extract(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 400:
        x = np.pad(x, (0, 400 - len(x)))
    _, _, z = stft(x, fs=SR, nperseg=400, noverlap=240, nfft=512, boundary=None)
    power = np.maximum(np.abs(z) ** 2, 1e-12)
    # Forty linearly spaced triangular filters over the full analysis band.
    bins = np.linspace(0, power.shape[0] - 1, 42)
    freq = np.arange(power.shape[0])
    filters = np.array([np.maximum(0, np.minimum((freq - a) / (b - a), (c - freq) / (c - b)))
                        for a, b, c in zip(bins[:-2], bins[1:-1], bins[2:])])
    cepstra = dct(np.log(filters @ power + 1e-12), type=2, axis=0, norm='ortho')[1:21]
    delta = np.diff(cepstra, axis=1) if cepstra.shape[1] > 1 else np.zeros_like(cepstra)
    p = power / power.sum(axis=0, keepdims=True)
    hz = np.linspace(0, SR / 2, len(freq))[:, None]
    centroid = (p * hz).sum(axis=0) / SR
    flatness = np.exp(np.mean(np.log(power), axis=0)) / np.mean(power, axis=0)
    entropy = -(p * np.log(p + 1e-12)).sum(axis=0) / np.log(len(freq))
    high = p[hz[:, 0] > 3400].sum(axis=0)
    summaries = [cepstra.mean(axis=1), cepstra.std(axis=1), delta.mean(axis=1), delta.std(axis=1)]
    for values in (centroid, flatness, entropy, high):
        summaries.append(np.array([values.mean(), values.std()]))
    return np.concatenate(summaries)


def quality(x):
    rms = float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0
    clipping = float(np.mean(np.abs(x) >= 0.99)) if len(x) else 0
    reasons = []
    if rms < 0.004:
        reasons.append('Audio level too low or silence')
    if clipping > 0.05:
        reasons.append('Severe clipping')
    return {'rms_dbfs': round(20 * np.log10(max(rms, 1e-8)), 1),
            'clipping_fraction': round(clipping, 4), 'usable': not reasons, 'reasons': reasons}

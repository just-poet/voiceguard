import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import expit

from .audio import HOP, SR, telephone
from .features import FEATURE_VERSION, extract, quality

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / 'models' / 'detector.json'


def prepare_head(head):
    runtime = dict(head)
    for name in ('mean','scale','coef','support_vectors','dual'):
        if name in runtime:
            runtime[name] = np.asarray(runtime[name],dtype=np.float64)
    return runtime


def predict_head(head, features):
    """Safe JSON inference: no pickle execution or sklearn runtime model loading."""
    scaled = (features - np.asarray(head['mean'])) / np.asarray(head['scale'])
    if head.get('kind') == 'rbf-svm':
        difference = np.asarray(head['support_vectors']) - scaled
        kernel = np.exp(-head['gamma'] * np.einsum('ij,ij->i',difference,difference))
        return float(expit(np.dot(kernel, head['dual']) + head['intercept']))
    if head.get('kind', 'linear') == 'extra-trees':
        scaled = scaled.astype(np.float32)
        probabilities = []
        for tree in head['trees']:
            node = 0
            while tree['left'][node] != -1:
                node = tree['left'][node] if scaled[tree['feature'][node]] <= tree['threshold'][node] else tree['right'][node]
            probabilities.append(tree['probability'][node])
        return float(np.mean(probabilities))
    return float(expit(np.dot(scaled, head['coef']) + head['intercept']))


class Detector:
    def __init__(self, model_path=DEFAULT_MODEL):
        self._prepared_model = None
        self.path = Path(model_path)
        self.model = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else None
        if self.model:
            if self.model.get('feature_version') != FEATURE_VERSION:
                raise ValueError('Model feature version mismatch. Retrain the detector.')
            for head in [self.model] + ([self.model['local_head']] if 'local_head' in self.model else []):
                fields = ('mean', 'scale') if head.get('kind') in ('extra-trees','rbf-svm') else ('mean','scale','coef')
                for key in fields:
                    values = np.asarray(head[key], dtype=float)
                    if values.shape != (88,) or not np.isfinite(values).all():
                        raise ValueError(f'Invalid model {key}.')
                if np.any(np.asarray(head['scale']) <= 0):
                    raise ValueError('Invalid model scales.')
                if head.get('kind') == 'extra-trees' and not head.get('trees'):
                    raise ValueError('Empty tree ensemble.')

    def status(self):
        return {'ready': self.model is not None, 'name': 'LFCC two-timescale + channel consistency',
                'classifier': self.model.get('kind', 'linear') if self.model else None,
                'feature_version': FEATURE_VERSION,
                'training': self.model.get('training') if self.model else None,
                'warning': 'Research baseline; scores are not calibrated probabilities or proof of fraud.'}

    def score_window(self, x):
        start = perf_counter()
        q = quality(x)
        result = {'score': None, 'context_score': None, 'label': 'untrained', 'quality': q,
                  'views': {}, 'context_views': {}, 'disagreement': None, 'context_disagreement': None}
        if self.model:
            if self._prepared_model is not self.model:
                self._context_head = prepare_head(self.model)
                self._local_head = prepare_head(self.model['local_head']) if self.model.get('local_head') else None
                self._prepared_model = self.model
            views = {'native': x, 'mulaw': telephone(x, 'mulaw'), 'alaw': telephone(x, 'alaw')}
            m = self.model
            weights = np.array([m.get('view_weights',{}).get(name,1/3) for name in views],dtype=float)
            weights /= weights.sum()
            values, local_values = [], []
            for name, y in views.items():
                s = predict_head(self._context_head, extract(y))
                values.append(s)
                result['context_views'][name] = round(s, 5)
                head = self._local_head
                if head:
                    local_score = predict_head(head, extract(y[-HOP:]))
                else:
                    local_score = s
                local_values.append(local_score)
                result['views'][name] = round(local_score, 5)
            score = float(np.dot(local_values,weights))
            spread = float(np.ptp(local_values))
            threshold = m.get('local_threshold', m['threshold'])
            label = 'synthetic-evidence' if score >= threshold else 'genuine-evidence'
            if not q['usable'] or spread > m.get('max_disagreement', 0.25) or abs(score - threshold) < m.get('margin', 0.08):
                label = 'uncertain'
            result.update(score=round(score, 5), context_score=round(float(np.dot(values,weights)), 5), label=label,
                          disagreement=round(spread, 5), context_disagreement=round(float(np.ptp(values)),5))
        elif not q['usable']:
            result['label'] = 'insufficient-audio'
        result['processing_ms'] = round((perf_counter() - start) * 1000, 2)
        return result

    def analyze(self, x):
        """Each bin uses only audio available up to its end (causal 2s context)."""
        bins = []
        for offset in range(0, len(x), HOP):
            end = min(offset + HOP, len(x))
            window = x[max(0, end - 2 * SR):end]
            r = self.score_window(window)
            if end < 2 * SR and self.model:
                r['label'] = 'warming-up'
                r.update(score=None, context_score=None, views={}, context_views={}, disagreement=None, context_disagreement=None)
            bins.append({'start': offset / SR, 'end': end / SR, **r})
        scores = [b['context_score'] for b in bins if b['context_score'] is not None and b['end'] >= 2]
        summary = {'score': None, 'label': 'untrained' if not self.model else 'insufficient-audio'}
        if scores:
            # Pool the largest two bins to retain short manipulation evidence.
            score = float(np.mean(sorted(scores)[-2:]))
            evidence_bins = sorted([b for b in bins if b['end'] >= 2], key=lambda b:b['context_score'])[-2:]
            label = 'synthetic-evidence' if score >= self.model['threshold'] else 'genuine-evidence'
            if (not quality(x)['usable'] or abs(score-self.model['threshold']) < self.model.get('margin',.08)
                    or any(b['context_disagreement'] > self.model.get('max_disagreement',.25) for b in evidence_bins)):
                label = 'uncertain'
            summary = {'score': round(score, 5), 'label': label,
                       'views': {name:round(float(np.mean([b['context_views'][name] for b in evidence_bins])),5)
                                 for name in ('native','mulaw','alaw')},
                       'disagreement': max(b['context_disagreement'] for b in evidence_bins)}
        wave = [round(float(np.max(np.abs(v))), 4) for v in np.array_split(x, min(160, len(x)))]
        return {'duration': round(len(x) / SR, 3), 'summary': summary, 'bins': bins,
                'waveform': wave, 'model': self.status(), 'quality': quality(x),
                'resolution_seconds': 0.5, 'context_seconds': 2, 'local_window_seconds': .5,
                'note': 'Timeline uses a 0.5s head; clip summary pools the top two 2s-context scores. First 2s are warm-up.'}

"""python -m callguard.cli --help"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from .audio import CONDITIONS, HOP, SR, condition, decode
from .dataset import check_evaluation_rows, read_manifest, validate_splits
from .detector import DEFAULT_MODEL, Detector
from .features import FEATURE_VERSION, extract
from .metrics import binary_metrics, eer


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def fit_head(features, labels, kind, seed):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    if kind == 'extra-trees':
        classifier = ExtraTreesClassifier(n_estimators=128, max_depth=14, min_samples_leaf=8,
                                          class_weight='balanced', n_jobs=4, random_state=seed)
    elif kind == 'rbf-svm':
        classifier = SVC(C=1, gamma='scale', class_weight='balanced', cache_size=512, random_state=seed)
    else:
        classifier = LogisticRegression(C=.1, class_weight='balanced', max_iter=2000, random_state=seed)
    fit_indices = np.arange(len(labels))
    if kind == 'rbf-svm' and len(labels) > 8192:
        target = np.asarray(labels)
        rng = np.random.default_rng(seed)
        fit_indices = np.concatenate([rng.choice(np.flatnonzero(target == label), min(4096,int(np.sum(target==label))),replace=False) for label in (0,1)])
    classifier.fit(scaled[fit_indices], np.asarray(labels)[fit_indices])
    head = {'kind':kind, 'mean':scaler.mean_.tolist(), 'scale':scaler.scale_.tolist(), 'fit_examples':len(fit_indices)}
    if kind == 'extra-trees':
        head['trees'] = []
        for estimator in classifier.estimators_:
            t = estimator.tree_
            values = t.value[:,0,:]
            probabilities = values[:,1] / values.sum(axis=1)
            head['trees'].append({'left':t.children_left.tolist(), 'right':t.children_right.tolist(),
                                  'feature':t.feature.tolist(), 'threshold':t.threshold.tolist(),
                                  'probability':probabilities.tolist()})
    elif kind == 'rbf-svm':
        head.update(support_vectors=classifier.support_vectors_.tolist(), dual=classifier.dual_coef_[0].tolist(),
                    gamma=float(classifier._gamma), intercept=float(classifier.intercept_[0]))
    else:
        head.update(coef=classifier.coef_[0].tolist(), intercept=float(classifier.intercept_[0]))
    return head


def train(args):
    rows = read_manifest(args.manifest)
    validate_splits(rows, exploratory=args.exploratory)
    if any('fake_segments' in r for r in rows):
        raise ValueError('Train on full genuine/synthetic clips; keep partials for localization testing.')
    features, local_features, labels = [], [], []
    training = [r for r in rows if r['split'] == 'train']
    cache_key = hashlib.sha256(json.dumps([FEATURE_VERSION, 'twohead-v1', args.seed, [(r['sha256'],r['label']) for r in training]]).encode()).hexdigest()[:20]
    cache_path = Path('data') / 'cache' / f'{cache_key}.npz'
    cached = cache_path.exists()
    if cached:
        with np.load(cache_path, allow_pickle=False) as saved:
            features, local_features, labels = saved['context'], saved['local'], saved['labels']
        print('Reusing cached training features.', flush=True)
    for i, row in enumerate([] if cached else training):
        x, _ = decode(row['path'])
        if len(x) < 2 * SR:
            raise ValueError('Training clips must be 2–10 seconds.')
        for name in ('clean', 'mulaw', 'alaw', 'mulaw_noise10', 'alaw_noise10'):
            y = condition(x, name, seed=args.seed + i)
            for end in range(2 * SR, len(y) + 1, HOP):
                features.append(extract(y[end - 2 * SR:end]))
                local_features.append(extract(y[end - HOP:end]))
                labels.append(row['label'])
        if (i + 1) % 10 == 0:
            print(f'Features: {i + 1}/{len(training)} clips', flush=True)
    if not cached:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, context=np.array(features), local=np.array(local_features), labels=np.array(labels))
    print(f'Fitting {args.classifier} context and local heads...', flush=True)
    head = fit_head(features, labels, args.classifier, args.seed)
    local_head = fit_head(local_features, labels, args.classifier, args.seed)
    development = [r for r in rows if r['split'] != 'test']
    model = {'feature_version': FEATURE_VERSION, **head,
             'threshold': 0.5, 'local_threshold': .5, 'margin': 0.08, 'max_disagreement': 0.25,
             'local_head': local_head,
             'training': {'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': args.seed,
                          'exploratory': args.exploratory,
                          'speaker_and_generator_holdout_verified': not args.exploratory,
                          'languages': sorted({r.get('language', 'unspecified') for r in training}),
                          'train_clips': len(training), 'train_windows_augmented': len(features),
                          'development_hashes': sorted({r['sha256'] for r in development}),
                          'development_text_groups': sorted({r['text_group_sha256'] for r in development if r.get('text_group_sha256')}),
                          'development_speakers': sorted({r['speaker'] for r in development}),
                          'train_generators': sorted({r['generator'] for r in training if r['label']}),
                          'notice': 'LFCC research baseline; no claim of real-world reliability.'}}
    # Select the threshold on held-out development clips using identical clip pooling.
    detector = Detector(model_path=Path(args.output).with_suffix('.absent'))
    detector.model = model
    dev_y, dev_s, dev_local_y, dev_local_s, dev_context_s = [], [], [], [], []
    for row in rows:
        if row['split'] == 'dev':
            x, _ = decode(row['path'])
            result = detector.analyze(x)
            score = result['summary']['score']
            if score is None:
                raise ValueError('Development clips must be at least 2 seconds.')
            dev_y.append(row['label']); dev_s.append(score)
            for b in result['bins']:
                if b['score'] is not None:
                    dev_local_y.append(row['label']); dev_local_s.append(b['score'])
                    dev_context_s.append(b['context_score'])
    calibration = eer(dev_y, dev_s)
    model['threshold'] = calibration['threshold']
    model['local_threshold'] = eer(dev_local_y, dev_local_s)['threshold']
    model['context_bin_threshold'] = eer(dev_local_y, dev_context_s)['threshold']
    model['training']['development_eer_percent'] = calibration['eer_percent']
    model['training']['threshold_source'] = 'development clips, top-two-bin pooling'
    write_json(args.output, model)
    print(f'Model saved: {Path(args.output).resolve()} — restart the server to load it.')


def evaluate(args):
    detector = Detector(args.model)
    if detector.model is None:
        raise ValueError('Train a model before evaluation.')
    rows = [r for r in read_manifest(args.manifest) if r['split'] == 'test']
    if args.partials:
        rows.extend(read_manifest(args.partials))
    if not rows:
        raise ValueError('No test rows.')
    check_evaluation_rows(rows, detector.model)
    report = {'model': detector.status(), 'conditions': {}, 'trials': [], 'test_clips': len(rows),
              'exploratory': detector.model['training'].get('exploratory', False),
              'notes': ['EER uses raw scores, including abstained clips; abstention coverage is reported separately.',
                        'Localization uses a separate 0.5s head, >=50% synthetic overlap labels and a fixed development-bin threshold.',
                        'Bins ending before 2s are warm-up, counted as genuine for localization; all bins remain in the denominator.',
                        'G.711 A-law/mu-law plus a 300–3400Hz bandpass simulate telephone effects, not real call capture.',
                        'CPU timing is measured on this computer; actual streaming may include additional browser/network latency.']}
    partial_count = sum('fake_segments' in r for r in rows)
    for name in CONDITIONS:
        ys, scores, native_scores, accepted, latency, bin_y, bin_p, bin_long, realtime_factors = [], [], [], [], [], [], [], [], []
        language_bins = {}
        for i, row in enumerate(rows):
            x, _ = decode(row['path'])
            begin = perf_counter()
            result = detector.analyze(condition(x, name, seed=args.seed + i))
            duration_ms = (perf_counter() - begin) * 1000
            realtime_factors.append(duration_ms / 1000 / (len(x) / SR))
            if result['summary']['score'] is None:
                raise ValueError('Test clips must be at least 2 seconds.')
            ys.append(row['label']); scores.append(result['summary']['score'])
            native_bins = [b['context_views']['native'] for b in result['bins'] if b['end'] >= 2]
            native_scores.append(float(np.mean(sorted(native_bins)[-2:])))
            accepted.append(result['summary']['label'] in ('genuine-evidence', 'synthetic-evidence'))
            latency.extend(b['processing_ms'] for b in result['bins'])
            report['trials'].append({'condition':name, 'file':Path(row['path']).name, 'sha256':row['sha256'],
                                     'language':row.get('language','unspecified'), 'label':row['label'],
                                     'partial':'fake_segments' in row, 'fake_segments':row.get('fake_segments',[]),
                                     'score':scores[-1], 'decision':result['summary']['label'],
                                     'bins':[{k:b[k] for k in ('start','end','score','context_score','label')} for b in result['bins']]})
            if 'fake_segments' in row:
                lang = row.get('language','unspecified')
                local_labels, local_predictions = language_bins.setdefault(lang, ([],[]))
                for b in result['bins']:
                    overlap = sum(max(0, min(b['end'], end) - max(b['start'], start)) for start, end in row['fake_segments'])
                    bin_y.append(overlap >= (b['end'] - b['start']) / 2)
                    bin_p.append(b['end'] >= 2 and b['score'] >= detector.model.get('local_threshold',detector.model['threshold']))
                    bin_long.append(b['end'] >= 2 and b['context_score'] >= detector.model.get('context_bin_threshold',detector.model['threshold']))
                    local_labels.append(bin_y[-1]); local_predictions.append(bin_p[-1])
        full = [i for i, r in enumerate(rows) if 'fake_segments' not in r]
        full_y = [ys[i] for i in full]; full_s = [scores[i] for i in full]
        entry = {'clip_eer': eer(full_y, full_s) if set(full_y) == {0, 1} else None,
                 'native_only_ablation_eer': eer(full_y, [native_scores[i] for i in full]) if set(full_y) == {0, 1} else None,
                 'full_clip_count': len(full), 'accepted_fraction': float(np.mean(accepted)),
                 'fixed_threshold_metrics': binary_metrics(ys, np.array(scores) >= detector.model['threshold']),
                 'bin_processing_p50_ms': float(np.median(latency)), 'bin_processing_p95_ms': float(np.percentile(latency, 95)),
                 'realtime_factor_mean': float(np.mean(realtime_factors)),
                 'localization': binary_metrics(bin_y, bin_p) if bin_y else None,
                 'localization_long_head_ablation': binary_metrics(bin_y, bin_long) if bin_y else None}
        accept_indices = [i for i,a in enumerate(accepted) if a]
        entry['accepted_set_error_fraction'] = float(np.mean([(scores[i]>=detector.model['threshold'])!=bool(ys[i]) for i in accept_indices])) if accept_indices else None
        entry['by_language'] = {}
        for language in sorted({r.get('language', 'unspecified') for r in rows}):
            indices = [i for i in full if rows[i].get('language', 'unspecified') == language]
            labels = [ys[i] for i in indices]
            entry['by_language'][language] = {'clips':len(indices), 'eer':eer(labels, [scores[i] for i in indices]) if set(labels)=={0,1} else None}
            if language in language_bins:
                entry['by_language'][language]['localization'] = binary_metrics(*language_bins[language])
        report['conditions'][name] = entry
        print(f'Evaluated {name}', flush=True)
    clean = report['conditions']['clean']['clip_eer']
    degraded = [v['clip_eer']['eer_percent'] for k, v in report['conditions'].items() if k != 'clean' and v['clip_eer']]
    worst = max(degraded) if degraded else None
    clean_eer = clean['eer_percent'] if clean else None
    phase1_delta = clean_eer - args.phase1_eer if args.phase1_eer is not None and clean_eer is not None else None
    report['phase2'] = {'clean_eer_percent': clean_eer, 'worst_degraded_eer_percent': worst,
                        'speaker_and_generator_holdout_verified': detector.model['training'].get('speaker_and_generator_holdout_verified', False),
                        'within_2_5x_clean': worst <= 2.5 * clean_eer if worst is not None and clean_eer is not None else None,
                        'degradation_ratio': worst / clean_eer if worst is not None and clean_eer else None,
                        'partial_clips': partial_count, 'at_least_100_partials': partial_count >= 100,
                        'same_speaker_partials_verified': all(r.get('same_speaker_verified', False) for r in rows if 'fake_segments' in r) if partial_count else None,
                        'phase1_eer_percent': args.phase1_eer, 'phase1_delta_points': phase1_delta,
                        'within_2_phase1_points': phase1_delta <= 2 if phase1_delta is not None else None,
                        'timing_under_500ms_all_conditions': all(v['bin_processing_p95_ms'] < 500 for v in report['conditions'].values()),
                        'phase1_comparison_caveat': 'Valid only with the same Phase 1 test set and protocol.'}
    write_json(args.output, report)
    print(json.dumps(report['phase2'], indent=2))


def partials(args):
    if args.count < 1:
        raise ValueError('Count must be positive.')
    rows = [r for r in read_manifest(args.manifest) if r['split'] == 'test' and 'fake_segments' not in r]
    real = [r for r in rows if r['label'] == 0]
    fake = [r for r in rows if r['label'] == 1]
    rng = np.random.default_rng(args.seed)
    pairs = [(a, b) for a in real for b in fake if (a['speaker'] == b['speaker'] and a['speaker'] != 'unknown')
             or (getattr(args, 'exploratory', False) and a.get('language') == b.get('language'))]
    if not pairs:
        raise ValueError('Need real and cloned test speech with matching speaker IDs. Cross-speaker splices confound the task.')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = []
    usable_pairs = []
    cache = {}
    for a, b in pairs:
        for row in (a, b):
            if row['path'] not in cache:
                cache[row['path']] = decode(row['path'])[0]
        if len(cache[a['path']]) >= 4 * SR and len(cache[b['path']]) >= SR:
            usable_pairs.append((a,b))
    if not usable_pairs:
        raise ValueError('No usable 4–10s genuine clips with synthetic donors.')
    rng.shuffle(usable_pairs)
    language_pairs = {}
    for pair in usable_pairs:
        language_pairs.setdefault(pair[0].get('language','unspecified'),[]).append(pair)
    languages = sorted(language_pairs)
    for i in range(args.count):
        group = language_pairs[languages[i % len(languages)]]
        a, b = group[(i // len(languages)) % len(group)]
        x = cache[a['path']].copy(); y = cache[b['path']]
        if len(x) < 4 * SR or len(y) < SR:
            raise ValueError('Partial-fake source needs genuine clips of 4–10s and synthetic donors of at least 1s.')
        length_bins = int(rng.integers(1, min(4, len(y) // HOP, (len(x) - 2 * SR) // HOP) + 1))
        count = length_bins * HOP
        start_bin = int(rng.integers(4, (len(x) - count) // HOP + 1))
        start = start_bin * HOP
        donor_start = int(rng.integers(0, len(y) - count + 1))
        donor = y[donor_start:donor_start + count].copy()
        target = x[start:start + count]
        donor *= min(3, np.sqrt(np.mean(target ** 2) + 1e-9) / np.sqrt(np.mean(donor ** 2) + 1e-9))
        envelope = np.ones(count)
        fade = 160  # 10 ms transitions; transition samples are labeled synthetic.
        envelope[:fade] = np.linspace(0, 1, fade)
        envelope[-fade:] = np.linspace(1, 0, fade)
        x[start:start + count] = target * (1 - envelope) + donor * envelope
        name = f'partial_{i:04d}.wav'
        sf.write(out / name, np.clip(x, -1, 1), SR, subtype='PCM_16')
        records.append({'path': name, 'label': 1, 'speaker': a['speaker'], 'generator': b['generator'], 'split': 'test',
                        'fake_segments': [[start / SR, (start + count) / SR]],
                        'source_hashes': [a['sha256'], b['sha256']],
                        'language': a.get('language', 'unspecified'),
                        'same_speaker_verified': a['speaker'] == b['speaker'] and a['speaker'] != 'unknown',
                        'source_real': a['path'], 'source_synthetic': b['path']})
    (out / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
    write_json(out / 'provenance.json', {'seed': args.seed, 'count': args.count, 'unique_source_pairs': min(args.count,len(usable_pairs)),
                                      'same_speaker_verified': all(r['same_speaker_verified'] for r in records),
                                      'note': 'Repeated source pairs are not independent examples. Splice transitions are included in labels.'})
    print(f'Wrote {args.count} partial clips: {out / "manifest.jsonl"}')


def fixtures(args):
    """Signal diagnostics only. Never presented as genuine or cloned speech."""
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t = np.arange(6 * SR) / SR
    voice_like = .18 * np.sin(2 * np.pi * (145 * t + 5 * np.sin(t))) * (.6 + .4 * np.sin(2 * np.pi * 3 * t) ** 2)
    splice = voice_like.copy()
    splice[3 * SR:4 * SR] = .18 * np.sin(2 * np.pi * 700 * t[3 * SR:4 * SR])
    for name, x in [('signal', voice_like), ('splice', splice), ('telephone', condition(voice_like, 'mulaw_noise10'))]:
        sf.write(out / f'{name}.wav', x, SR, subtype='PCM_16')
    print(f'Created diagnostic signals in {out.resolve()}. These are not a speech benchmark.')


def main():
    p = argparse.ArgumentParser(description='CallGuard — free local speech authenticity research tools')
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train'); t.add_argument('manifest'); t.add_argument('--exploratory', action='store_true'); t.add_argument('--classifier', choices=['linear','extra-trees','rbf-svm'], default='linear'); t.add_argument('--output', default=str(DEFAULT_MODEL)); t.add_argument('--seed', type=int, default=42); t.set_defaults(func=train)
    e = sub.add_parser('evaluate'); e.add_argument('manifest'); e.add_argument('--partials'); e.add_argument('--model', default=str(DEFAULT_MODEL)); e.add_argument('--output', default='reports/evaluation.json'); e.add_argument('--phase1-eer', type=float); e.add_argument('--seed', type=int, default=42); e.set_defaults(func=evaluate)
    g = sub.add_parser('partials'); g.add_argument('manifest'); g.add_argument('--exploratory', action='store_true'); g.add_argument('--output', default='data/partials'); g.add_argument('--count', type=int, default=100); g.add_argument('--seed', type=int, default=42); g.set_defaults(func=partials)
    d = sub.add_parser('fixtures'); d.add_argument('--output', default='data/diagnostics'); d.set_defaults(func=fixtures)
    args = p.parse_args()
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        p.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()

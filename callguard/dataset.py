"""Explicit manifests and leakage checks; no automatic random file splitting."""
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_manifest(path):
    path = Path(path).resolve()
    rows = []
    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {'path', 'label', 'speaker', 'generator', 'split'}
        if not required <= row.keys():
            raise ValueError(f'Line {number}: missing {required - row.keys()}')
        if type(row['label']) is not int or row['label'] not in (0, 1) or row['split'] not in ('train', 'dev', 'test'):
            raise ValueError(f'Line {number}: label must be 0/1 and split train/dev/test.')
        if not row['speaker'] or not row['generator']:
            raise ValueError(f'Line {number}: speaker and generator cannot be empty.')
        row['path'] = str((path.parent / row['path']).resolve())
        if not Path(row['path']).is_file():
            raise ValueError(f'Line {number}: file missing: {row["path"]}')
        row['sha256'] = digest(row['path'])
        if 'fake_segments' in row:
            segments = row['fake_segments']
            if row['label'] != 1 or not isinstance(segments, list) or not segments:
                raise ValueError(f'Line {number}: partial clips need label 1 and nonempty segments.')
            previous = 0
            for segment in segments:
                if not isinstance(segment, list) or len(segment) != 2:
                    raise ValueError(f'Line {number}: use [start_seconds, end_seconds] segments.')
                start, end = segment
                if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or not 0 <= previous <= start < end <= 10:
                    raise ValueError(f'Line {number}: invalid or overlapping fake segments.')
                previous = end
        rows.append(row)
    if not rows:
        raise ValueError('Manifest is empty.')
    return rows


def validate_splits(rows, exploratory=False):
    for split in ('train', 'dev', 'test'):
        subset = [r for r in rows if r['split'] == split]
        if {r['label'] for r in subset} != {0, 1}:
            raise ValueError(f'{split} needs both genuine (0) and synthetic (1) examples.')
    for field in ('speaker', 'sha256'):
        seen = {}
        for row in rows:
            value = row[field]
            if field == 'speaker' and value == 'unknown' and exploratory:
                continue
            if field == 'speaker' and value == 'unknown':
                raise ValueError('Unknown speakers require --exploratory; speaker holdout cannot be verified.')
            if value in seen and seen[value] != row['split']:
                raise ValueError(f'{field} leaks across splits: {value}')
            seen[value] = row['split']
    text_groups = {}
    for row in rows:
        group = row.get('text_group_sha256')
        if group:
            if group in text_groups and text_groups[group] != row['split']:
                raise ValueError('Transcript text group leaks across splits.')
            text_groups[group] = row['split']
    trained = {r['generator'] for r in rows if r['label'] and r['split'] == 'train'}
    tested = {r['generator'] for r in rows if r['label'] and r['split'] == 'test'}
    if exploratory:
        trained.discard('unknown'); tested.discard('unknown')
    elif 'unknown' in trained | tested:
        raise ValueError('Unknown synthesis tools require --exploratory; tool holdout cannot be verified.')
    if trained & tested:
        raise ValueError(f'Hold out test synthesis tools entirely; overlap: {trained & tested}')


def check_evaluation_rows(rows, model):
    history = model['training']
    hashes = set(history['development_hashes'])
    speakers = set(history['development_speakers'])
    generators = set(history['train_generators'])
    text_groups = set(history.get('development_text_groups', []))
    for r in rows:
        if r['split'] != 'test':
            raise ValueError('Only test rows may be evaluated.')
        provenance = r.get('source_hashes', [r['sha256']])
        if r.get('text_group_sha256') in text_groups:
            raise ValueError('Evaluation transcript text overlaps development data.')
        speaker_overlap = r['speaker'] in speakers and not (history.get('exploratory') and r['speaker'] == 'unknown')
        if hashes.intersection(provenance) or r['sha256'] in hashes or speaker_overlap:
            raise ValueError('Evaluation data overlaps training/development sources or speakers.')
        unknown_tool = history.get('exploratory') and r['generator'] == 'unknown'
        if r['label'] and r['generator'] in generators and not unknown_tool:
            raise ValueError('Evaluation synthesis tool was used in training.')

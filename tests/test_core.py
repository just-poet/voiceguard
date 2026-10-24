from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from callguard.audio import HOP, SR, _linear_to_g711, condition, decode
from callguard.cli import partials
from callguard.dataset import check_evaluation_rows, read_manifest, validate_splits
from callguard.detector import Detector
from callguard.detector import predict_head
from callguard.cli import fit_head
from callguard.features import FEATURE_VERSION, extract
from callguard.metrics import eer
from callguard import server


def wav(x, rate=SR):
    buffer = BytesIO(); sf.write(buffer, x, rate, format='WAV', subtype='PCM_16')
    return buffer.getvalue()


def signal(seconds=4):
    return .15 * np.sin(2 * np.pi * 173 * np.arange(int(seconds * SR)) / SR)


def test_decode_bounds_and_8k():
    x, rate = decode(wav(signal(2)[::2], 8000))
    assert rate == 8000 and len(x) == 2 * SR
    with pytest.raises(ValueError, match='at most'):
        decode(wav(signal(11)))
    with pytest.raises(ValueError):
        decode(b'not an audio file')


def test_g711_known_zero_and_full_scale():
    u = _linear_to_g711(np.array([-1., 0., 1.]), 'mulaw')
    a = _linear_to_g711(np.array([-1., 0., 1.]), 'alaw')
    np.testing.assert_allclose(u, [-32124/32768, 0, 32124/32768])
    np.testing.assert_allclose(a, [-32256/32768, 8/32768, 32256/32768])


def test_channels_are_deterministic_finite_length_preserving():
    x = signal()
    for name in ('resample8k', 'mulaw', 'alaw', 'mulaw_noise10', 'alaw_noise10'):
        y = condition(x, name)
        assert len(x) == len(y) and np.isfinite(y).all()
        np.testing.assert_array_equal(y, condition(x, name))
    assert not np.array_equal(condition(x, 'mulaw'), condition(x, 'alaw'))


def test_eer_ties_perfect_and_inverted():
    assert eer([0, 0, 1, 1], [.1, .2, .8, .9])['eer_percent'] == 0
    assert eer([0, 1], [.5, .5])['eer_percent'] == 50
    assert eer([0, 1], [.9, .1])['eer_percent'] == 100
    with pytest.raises(ValueError): eer([1, 1], [.1, .2])


def test_untrained_never_invents_scores(tmp_path):
    d = Detector(tmp_path / 'absent.json')
    r = d.analyze(signal())
    assert r['summary']['score'] is None
    assert len(r['bins']) == 8
    assert all(b['score'] is None and not b['views'] for b in r['bins'])
    assert d.score_window(np.zeros(SR))['label'] == 'insufficient-audio'


def test_features_and_causal_windows(tmp_path):
    d = Detector(tmp_path / 'absent.json')
    f = extract(signal(2))
    assert f.shape == (88,) and np.isfinite(f).all()
    d.model = {'mean':[0]*88,'scale':[1]*88,'coef':[.001]*88,'intercept':0,'threshold':.5,'max_disagreement':.25,'margin':.08}
    x = signal()
    offline = d.analyze(x)
    for i, b in enumerate(offline['bins']):
        end = (i + 1) * HOP
        causal = d.score_window(x[max(0, end - 2*SR):end])
        if end >= 2 * SR:
            assert causal['score'] == b['score']
        else:
            assert b['score'] is None and b['label'] == 'warming-up'
    changed = x.copy(); changed[3*SR:] = .3
    # Compare evidence only; wall-clock timing naturally differs.
    assert [b['score'] for b in d.analyze(changed)['bins'][:6]] == [b['score'] for b in offline['bins'][:6]]


def rows_fixture():
    return [{'path':f'{split}-{label}', 'split':split, 'label':label, 'speaker':split,
             'sha256':f'{split}-{label}', 'generator': ('heldout' if split=='test' else 'known') if label else 'human'}
            for split in ('train', 'dev', 'test') for label in (0, 1)]


def test_leakage_guards():
    rows = rows_fixture(); validate_splits(rows)
    rows[-1]['speaker'] = 'train'
    with pytest.raises(ValueError, match='speaker'): validate_splits(rows)
    rows = rows_fixture(); rows[-1]['generator'] = 'known'
    with pytest.raises(ValueError, match='synthesis'): validate_splits(rows)
    model = {'training':{'development_hashes':['source-A'], 'development_speakers':['train'], 'train_generators':['known']}}
    test = rows_fixture()[-1]; test['source_hashes']=['source-A']
    with pytest.raises(ValueError, match='overlaps'): check_evaluation_rows([test], model)


def test_exploratory_mode_retains_content_and_text_guards():
    rows=rows_fixture()
    for row in rows: row['speaker']='unknown';row['generator']='unknown' if row['label'] else 'human'
    with pytest.raises(ValueError,match='Unknown speakers'):validate_splits(rows)
    validate_splits(rows,exploratory=True)
    rows[0]['text_group_sha256']='same-text';rows[-1]['text_group_sha256']='same-text'
    with pytest.raises(ValueError,match='text group'):validate_splits(rows,exploratory=True)


def test_tree_json_matches_sklearn():
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.preprocessing import StandardScaler
    rng=np.random.default_rng(9)
    features=rng.normal(size=(100,88));labels=(features[:,0]+features[:,1]>0).astype(int)
    head=fit_head(features,labels,'extra-trees',42)
    scaled=StandardScaler().fit_transform(features)
    reference=ExtraTreesClassifier(n_estimators=128,max_depth=14,min_samples_leaf=8,class_weight='balanced',n_jobs=4,random_state=42).fit(scaled,labels)
    observed=np.array([predict_head(head,f) for f in features[:12]])
    np.testing.assert_allclose(observed,reference.predict_proba(scaled[:12])[:,1],atol=1e-12)


def test_svm_json_matches_sklearn():
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from scipy.special import expit
    rng=np.random.default_rng(3)
    features=rng.normal(size=(80,88));labels=(features[:,0]>0).astype(int)
    head=fit_head(features,labels,'rbf-svm',42)
    scaled=StandardScaler().fit_transform(features)
    reference=SVC(C=1,gamma='scale',class_weight='balanced',cache_size=512,random_state=42).fit(scaled,labels)
    observed=np.array([predict_head(head,f) for f in features[:10]])
    np.testing.assert_allclose(observed,expit(reference.decision_function(scaled[:10])),atol=1e-10)


def test_partial_generation_100_and_labels(tmp_path):
    sf.write(tmp_path/'real.wav', signal(6), SR)
    sf.write(tmp_path/'fake.wav', signal(6)*.7, SR)
    rows=[{'path':name,'label':label,'speaker':'same-person','generator':generator,'split':'test'}
          for name,label,generator in [('real.wav',0,'human'),('fake.wav',1,'heldout-tool')]]
    manifest=tmp_path/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    out=tmp_path/'partials'
    partials(SimpleNamespace(manifest=str(manifest),output=str(out),count=100,seed=42))
    generated=read_manifest(out/'manifest.jsonl')
    assert len(generated)==100
    assert all(r['fake_segments'][0][0]>=2 for r in generated)
    assert all(len(r['source_hashes'])==2 for r in generated)


@pytest.fixture
def client(monkeypatch,tmp_path):
    monkeypatch.setattr(server,'detector',Detector(tmp_path/'absent.json'))
    with TestClient(server.app) as c: yield c


def test_api_audio_and_errors(client):
    assert client.get('/').status_code==200
    assert client.get('/api/status').json()['model']['ready'] is False
    r=client.post('/api/analyze',content=wav(signal(2)))
    assert r.status_code==200 and r.json()['summary']['score'] is None
    assert client.post('/api/analyze',content=b'invalid').status_code==422
    assert client.post('/api/analyze',content=wav(signal(1))).status_code==422
    assert client.post('/api/analyze',content=b'x'*(8*1024*1024+1)).status_code==413
    assert client.get('/api/diagnostic/splice').headers['content-type']=='audio/wav'
    assert client.get('/api/diagnostic/unknown').status_code==404


def test_stream_timestamps_and_bad_frames(client):
    with client.websocket_connect('/api/stream',headers={'origin':'http://127.0.0.1:8765'}) as ws:
        assert ws.receive_json()['type']=='ready'
        for i in range(4):
            ws.send_bytes(signal(.5).astype('<f4').tobytes())
            r=ws.receive_json()
            assert r['end']==(i+1)*.5 and r['score'] is None
    with client.websocket_connect('/api/stream',headers={'origin':'http://127.0.0.1:8765'}) as ws:
        ws.receive_json();ws.send_bytes(b'bad')
        assert ws.receive_json()['type']=='error'


def test_stream_rejects_external_origin(client):
    with pytest.raises(Exception):
        with client.websocket_connect('/api/stream',headers={'origin':'https://unrelated.example'}): pass

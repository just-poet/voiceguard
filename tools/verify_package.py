"""Check the actual ZIP using a fresh dependency environment and local HTTP."""
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
target = ROOT / 'tmp' / 'package-check'
target.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ROOT / 'output/CallGuard_Submission.zip') as archive:
    for member in archive.namelist():
        if not (target / member).resolve().is_relative_to(target.resolve()):
            raise ValueError('Unsafe archive path')
    archive.extractall(target)
python = ROOT / '.venv/Scripts/python.exe'
process = subprocess.Popen([str(python), '-m', 'uvicorn', 'callguard.server:app', '--host', '127.0.0.1', '--port', '8766'], cwd=target, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def get(path):
    return urllib.request.urlopen('http://127.0.0.1:8766' + path, timeout=20)
try:
    for attempt in range(60):
        try:
            status = json.load(get('/api/status'))
            break
        except OSError:
            if process.poll() is not None:
                raise RuntimeError('Packaged server exited')
            time.sleep(.5)
    else:
        raise RuntimeError('Packaged server did not start')
    assert status['model']['ready']
    assert b'CallGuard' in get('/').read()
    examples = json.load(get('/api/examples'))
    assert len(examples) == 6
    outcomes = []
    for example in examples:
        audio = get('/api/examples/' + example['id']).read()
        request = urllib.request.Request('http://127.0.0.1:8766/api/analyze', data=audio, headers={'Content-Type':'application/octet-stream'})
        result = json.load(urllib.request.urlopen(request, timeout=30))
        assert result['summary']['score'] is not None and len(result['bins']) >= 4
        outcomes.append({'example':example['id'], 'score':result['summary']['score']})
    assert len(list((target / 'data/partials').glob('*.wav'))) == 100
    result = {'passed':True, 'fresh_environment':str(python), 'demo_examples':outcomes, 'partial_clips':100, 'full_dataset_required_to_run':False}
    (ROOT / 'reports/package-verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
finally:
    process.terminate()
    process.wait(timeout=15)

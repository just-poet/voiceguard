import asyncio
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .audio import HOP, SR, condition, decode
from .detector import DEFAULT_MODEL, Detector

app = FastAPI(title='CallGuard', docs_url='/api/docs', redoc_url=None)
STATIC = Path(__file__).resolve().parents[1] / 'web'
ROOT = STATIC.parent
detector = Detector(DEFAULT_MODEL)


def public_examples():
    manifest = ROOT / 'data' / 'indictts' / 'manifest.jsonl'
    if not manifest.exists():
        manifest = ROOT / 'data' / 'demo' / 'manifest.jsonl'
    if not manifest.exists():
        return {}
    examples = {}
    for line in manifest.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row['split'] != 'test':
            continue
        key = row['language'].lower() + ('-synthetic' if row['label'] else '-genuine')
        if key not in examples:
            examples[key] = {'id': key, 'language': row['language'], 'label': row['label'],
                             'path': str((manifest.parent / row['path']).resolve()),
                             'source': 'IndicTTS challenge, held-out text group, CC BY 4.0'}
    return examples


@app.get('/')
def index():
    return FileResponse(STATIC / 'index.html')


@app.get('/api/status')
def status():
    return {'model': detector.status(), 'sample_rate': SR, 'hop_seconds': .5, 'local_only': True}


@app.get('/api/examples')
def list_examples():
    return [{k: v for k, v in row.items() if k != 'path'} for row in public_examples().values()]


@app.get('/api/examples/{example_id}')
def get_example(example_id: str):
    row = public_examples().get(example_id)
    if row is None:
        raise HTTPException(404, 'Unknown public example.')
    return FileResponse(row['path'], media_type='audio/wav')


@app.get('/api/evaluation')
def evaluation_report():
    path = ROOT / 'reports' / 'evaluation.json'
    if not path.exists():
        return {'available': False}
    report = json.loads(path.read_text(encoding='utf-8'))
    return {'available': True, 'conditions': report['conditions'], 'phase2': report['phase2'],
            'test_clips': report['test_clips'], 'exploratory': report.get('exploratory', True)}


@app.post('/api/analyze')
async def analyze(request: Request):
    payload = bytearray()
    async for chunk in request.stream():
        payload.extend(chunk)
        if len(payload) > 8 * 1024 * 1024:
            raise HTTPException(413, 'Maximum upload is 8 MB.')
    try:
        x, source_rate = await asyncio.to_thread(decode, bytes(payload))
        if len(x) < 2 * SR:
            raise ValueError('Choose a 2–10 second speech clip.')
        result = await asyncio.to_thread(detector.analyze, x)
        result['source_sample_rate'] = source_rate
        return result
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.get('/api/diagnostic/{name}')
def diagnostic(name: str):
    if name not in ('signal', 'splice', 'telephone'):
        raise HTTPException(404, 'Unknown diagnostic.')
    t = np.arange(6 * SR) / SR
    x = .18 * np.sin(2 * np.pi * (145 * t + 5 * np.sin(t))) * (.6 + .4 * np.sin(2 * np.pi * 3 * t) ** 2)
    if name == 'splice':
        x[3 * SR:4 * SR] = .18 * np.sin(2 * np.pi * 700 * t[3 * SR:4 * SR])
    if name == 'telephone':
        x = condition(x, 'mulaw_noise10')
    buffer = BytesIO()
    sf.write(buffer, x, SR, format='WAV', subtype='PCM_16')
    return Response(buffer.getvalue(), media_type='audio/wav', headers={'Cache-Control': 'no-store'})


@app.websocket('/api/stream')
async def stream(websocket: WebSocket):
    origin = websocket.headers.get('origin', '')
    if origin not in ('http://127.0.0.1:8765', 'http://localhost:8765'):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    history = np.array([], dtype=np.float64)
    samples = 0
    try:
        await websocket.send_json({'type': 'ready', 'model': detector.status(), 'sample_rate': SR, 'chunk_samples': HOP})
        while samples < SR * 300:
            payload = await asyncio.wait_for(websocket.receive_bytes(), timeout=30)
            if len(payload) != HOP * 4:
                await websocket.send_json({'type': 'error', 'message': 'Expected 8000 little-endian float32 samples per 500ms chunk.'})
                await websocket.close(code=1003)
                return
            x = np.frombuffer(payload, dtype='<f4').astype(np.float64)
            if not np.isfinite(x).all() or np.max(np.abs(x)) > 1.01:
                await websocket.close(code=1003)
                return
            history = np.concatenate((history, x))[-2 * SR:]
            samples += HOP
            result = await asyncio.to_thread(detector.score_window, history)
            if samples < 2 * SR and detector.model:
                result['label'] = 'warming-up'
                result.update(score=None, context_score=None, views={}, context_views={}, disagreement=None, context_disagreement=None)
            await websocket.send_json({'type': 'score', 'start': (samples - HOP) / SR, 'end': samples / SR, **result})
        await websocket.send_json({'type': 'end', 'message': 'Five-minute session limit reached.'})
        await websocket.close(code=1000)
    except (WebSocketDisconnect, asyncio.TimeoutError):
        return


app.mount('/static', StaticFiles(directory=STATIC), name='static')

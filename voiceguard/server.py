import asyncio,json,os,platform,struct,time
from pathlib import Path
import numpy as np
from fastapi import FastAPI,Request,HTTPException,WebSocket,WebSocketDisconnect
from fastapi.responses import FileResponse,Response
from fastapi.staticfiles import StaticFiles
from .model import Engine,ROOT,SR,HOP,N_SAMPLES
from .audio import decode,MAX_BYTES,ffmpeg_path

app=FastAPI(title='VoiceGuard',version='0.2.0',docs_url='/api/docs');engine=Engine();busy=asyncio.Semaphore(1)
DATA_ROOT=Path(os.environ.get('VOICEGUARD_DATA','D:/VoiceGuardData'))
ALLOWED_ORIGINS=set(os.environ.get('VOICEGUARD_ORIGINS','http://127.0.0.1:8765,http://localhost:8765,http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:8767').split(','))

@app.middleware('http')
async def local_only(request:Request,call_next):
    origin=request.headers.get('origin')
    if origin and origin not in ALLOWED_ORIGINS:return Response('Origin not allowed',status_code=403)
    response=await call_next(request);response.headers['Cache-Control']='no-store';return response

@app.get('/api/status')
def status():
    return {'app':'VoiceGuard','version':'0.2.0','model':engine.status(),'hardware':{'processor':platform.processor(),'platform':platform.platform(),'device':'CPU','threads':engine.status()['threads']},'ffmpeg_available':Path(ffmpeg_path()).is_file(),'raw_audio_retained':False,'history_enabled':False,'max_audio_seconds':10}

def examples():
    records=[];manifest=ROOT/'data/demo/manifest.jsonl'
    if manifest.exists():
        for i,line in enumerate(manifest.read_text().splitlines()):
            r=json.loads(line);records.append({'id':f'demo-{i}','name':r['language']+' · '+('synthetic' if r['label'] else 'bona fide'),'language':r['language'],'label':r['label'],'path':str(manifest.parent/r['path']),'kind':'full','source':'IndicTTS held-out example; CC BY 4.0'})
    manifest=DATA_ROOT/'partials/manifest.jsonl'
    if not manifest.exists():manifest=ROOT/'data/partials/manifest.jsonl'
    if manifest.exists():
        used=set()
        for line in manifest.read_text().splitlines():
            r=json.loads(line)
            if r['control_type'] in used:continue
            used.add(r['control_type']);records.append({'id':f'partial-{len(used)}','name':r['control_type'].replace('_',' '),'language':r.get('language'),'label':r['label'],'path':str(manifest.parent/r['path']),'kind':r['control_type'],'fake_segments':r['fake_segments'],'source':'Constructed evaluation proxy'})
    return records

@app.get('/api/examples')
def list_examples():return [{k:v for k,v in r.items() if k!='path'} for r in examples()]

@app.get('/api/examples/{example_id}')
def example(example_id:str):
    found=next((r for r in examples() if r['id']==example_id),None)
    if not found:raise HTTPException(404,'Unknown example')
    return FileResponse(found['path'],media_type='audio/wav')

@app.get('/api/evaluation')
def evaluation():
    output={'available':False,'benchmarks':{'ASVspoof2019-LA':'not evaluated','In-the-Wild':'not evaluated'},'downloads':[]}
    path=ROOT/'reports/voiceguard/evaluation.json'
    if path.exists():output.update(json.loads(path.read_text()));output['available']=True
    for name in ('asv_train_parquet','asv_dev_parquet','asv_test_parquet','in_the_wild'):
        p=DATA_ROOT/'archives'/f'{name}.progress.json'
        if p.exists():
            progress=json.loads(p.read_text());receipt=DATA_ROOT/'archives'/f'{name}.provenance.json'
            progress['archive_verified']=receipt.exists() and json.loads(receipt.read_text()).get('complete_archive',False)
            output['downloads'].append(progress)
    return output

@app.post('/api/analyze')
async def analyze(request:Request,model:str='candidate'):
    if busy.locked():raise HTTPException(429,'Engine busy. Stop the live session or wait for the current analysis.')
    payload=bytearray()
    async for block in request.stream():
        if len(payload)+len(block)>MAX_BYTES:raise HTTPException(413,'Maximum upload is 8 MB')
        payload.extend(block)
    try:
        if busy.locked():raise HTTPException(429,'Engine busy. Retry after the active analysis.')
        async with busy:
            x=await asyncio.to_thread(decode,bytes(payload))
            if len(x)<2*SR:raise ValueError('Choose a 2–10 second audio clip.')
            result=await asyncio.to_thread(engine.analyze,x,model)
            result['raw_audio_retained']=False;return result
    except ValueError as error:raise HTTPException(422,str(error)) from error

@app.post('/api/replay-pcm')
async def replay_pcm(request:Request):
    payload=bytearray()
    async for block in request.stream():
        if len(payload)+len(block)>MAX_BYTES:raise HTTPException(413,'Maximum upload is 8 MB')
        payload.extend(block)
    try:
        x=await asyncio.to_thread(decode,bytes(payload))
        if len(x)<2*SR:raise ValueError('Choose a 2–10 second clip')
        # Decoding only: no full-recording predictions are used for replay.
        return Response(x.astype('<f4').tobytes(),media_type='application/octet-stream',headers={'X-Sample-Rate':str(SR),'X-Frames':str(len(x))})
    except ValueError as error:raise HTTPException(422,str(error)) from error

def parse_packet(payload,expected):
    if len(payload)!=12+HOP*4:raise ValueError('Expected timestamp header plus 8000 float32 samples')
    sequence,start_sample=struct.unpack('<Id',payload[:12])
    if sequence!=expected or start_sample!=expected*HOP:raise ValueError('Out-of-order sequence or timestamp')
    x=np.frombuffer(payload[12:],dtype='<f4').astype(np.float64)
    if not np.isfinite(x).all() or np.max(np.abs(x))>1.01:raise ValueError('Invalid PCM samples')
    return x

@app.websocket('/api/stream')
async def stream(ws:WebSocket,model:str='candidate'):
    if ws.headers.get('origin') not in ALLOWED_ORIGINS or busy.locked():await ws.close(code=1008);return
    if model not in engine.status()['engines']:await ws.close(code=1008);return
    await ws.accept()
    if busy.locked():await ws.close(code=1008);return
    async with busy:
        history=np.array([],dtype=np.float64);sequence=0;last=None
        try:
            await ws.send_json({'type':'ready','sample_rate':SR,'chunk_samples':HOP,'warmup_seconds':2,'context_seconds':N_SAMPLES/SR,'max_pending_chunks':2,'max_session_seconds':300})
            while sequence<600:
                payload=await asyncio.wait_for(ws.receive_bytes(),timeout=30)
                arrival=time.perf_counter();x=parse_packet(payload,sequence)
                interval=None if last is None else (arrival-last)*1000;last=arrival
                history=np.concatenate([history,x])[-N_SAMPLES:]
                result=await asyncio.to_thread(engine.score,history,model)
                duration=(time.perf_counter()-arrival)*1000
                await ws.send_json({'type':'score','sequence':sequence,'start':sequence*.5,'end':(sequence+1)*.5,'arrival_interval_ms':interval,'server_latency_ms':duration,'realtime_factor':duration/500,'overloaded':duration>500,'waveform':[float(np.max(np.abs(v))) for v in np.array_split(x,12)],**result})
                sequence+=1
            await ws.send_json({'type':'end','message':'Five-minute limit reached'});await ws.close()
        except (WebSocketDisconnect,asyncio.TimeoutError):pass
        except ValueError as error:
            await ws.send_json({'type':'error','message':str(error)});await ws.close(code=1003)

DIST=ROOT/'frontend/dist'
if DIST.exists():app.mount('/',StaticFiles(directory=DIST,html=True),name='frontend')
else:
    @app.get('/')
    def setup():return {'message':'Build frontend: cd frontend && npm ci && npm run build'}

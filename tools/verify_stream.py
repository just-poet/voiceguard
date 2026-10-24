"""Verify the running API's real-time stream against its offline output."""
import asyncio
import json
from pathlib import Path
import sys
from time import perf_counter
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from websockets.asyncio.client import connect
from callguard.audio import SR,HOP,decode
from callguard.dataset import read_manifest


async def main():
    rows=read_manifest('data/indictts/manifest.jsonl')
    for row in rows:
        if row['split']=='test':
            x,_=decode(row['path'])
            if len(x)==6*SR:break
    else:raise ValueError('Need a six-second test clip for exact packet alignment.')
    request=urllib.request.Request('http://127.0.0.1:8765/api/analyze',data=Path(row['path']).read_bytes(),headers={'Content-Type':'application/octet-stream'})
    offline=json.load(urllib.request.urlopen(request,timeout=60))
    timings=[];arrivals=[];differences=[]
    async with connect('ws://127.0.0.1:8765/api/stream',origin='http://127.0.0.1:8765') as ws:
        assert json.loads(await ws.recv())['type']=='ready'
        start=perf_counter()
        for i,offset in enumerate(range(0,len(x),HOP)):
            await asyncio.sleep(max(0,start+(i+1)*.5-perf_counter()))
            before=perf_counter();await ws.send(x[offset:offset+HOP].astype('<f4').tobytes())
            live=json.loads(await ws.recv());after=perf_counter()
            timings.append((after-before)*1000);arrivals.append((after-start)*1000)
            assert live['start']==offline['bins'][i]['start'] and live['end']==offline['bins'][i]['end']
            expected=offline['bins'][i]['score']
            if expected is None:assert live['score'] is None
            else:differences.append(abs(live['score']-expected))
    assert max(differences)<1e-5,'Streaming scores differ from offline analysis.'
    report={'source_file':Path(row['path']).name,'language':row['language'],'duration_seconds':6,'chunks':len(timings),
            'chunk_ms':500,'warmup_seconds':2,'max_score_difference':max(differences),
            'roundtrip_p50_ms':float(np.median(timings)),'roundtrip_p95_ms':float(np.percentile(timings,95)),
            'roundtrip_max_ms':max(timings),'arrival_interval_p50_ms':float(np.median(np.diff(arrivals))),
            'all_responses_under_500ms':max(timings)<500,'raw_roundtrip_ms':timings,
            'note':'Local WebSocket replay with real wall-clock 500ms scheduling. This does not validate microphone hardware or cellular capture.'}
    Path('reports').mkdir(exist_ok=True)
    Path('reports/streaming-verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':asyncio.run(main())

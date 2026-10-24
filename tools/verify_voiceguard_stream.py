"""Timed real-checkpoint WebSocket replay, parity, malformed-packet checks."""
import asyncio,json,struct,sys,time,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from websockets.asyncio.client import connect
from voiceguard.audio import decode,SR,HOP

async def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765);p.add_argument('--model',default='candidate');a=p.parse_args();base=f'http://127.0.0.1:{a.port}'
    source=Path('D:/VoiceGuardData/partials/partial_0000.wav');data=source.read_bytes();x=decode(data)
    request=urllib.request.Request(base+'/api/analyze?model='+a.model,data=data,headers={'Content-Type':'application/octet-stream'})
    offline=json.load(urllib.request.urlopen(request,timeout=60));latencies=[];intervals=[];scores=[];previous=None
    async with connect(f'ws://127.0.0.1:{a.port}/api/stream?model={a.model}',origin=base,max_size=1024*1024) as ws:
        assert json.loads(await ws.recv())['type']=='ready';start=time.perf_counter()
        for i,offset in enumerate(range(0,len(x),HOP)):
            await asyncio.sleep(max(0,start+(i+1)*.5-time.perf_counter()))
            packet=struct.pack('<Id',i,offset)+x[offset:offset+HOP].astype('<f4').tobytes();before=time.perf_counter();await ws.send(packet);r=json.loads(await ws.recv());after=time.perf_counter()
            assert r['sequence']==i and r['start']==i*.5 and r['end']==(i+1)*.5
            expected=offline['bins'][i]['score']
            assert (r['score'] is None)==(expected is None)
            if expected is not None:assert abs(r['score']-expected)<1e-4
            latencies.append((after-before)*1000);scores.append(r)
            if r['score'] is not None:
                if previous is not None:intervals.append((after-previous)*1000)
                previous=after
        await ws.send(b'invalid');assert json.loads(await ws.recv())['type']=='error'
    all_latencies=latencies[:];latencies=[ms for ms,r in zip(latencies,scores) if r['score'] is not None]
    result={'backend':offline['model'].get('backend','pytorch'),'model_sha256':offline['model']['candidate_sha256'],'encoder_sha256':offline['model']['baseline_sha256'],'recorded_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'latency_scope':'Scored updates after warm-up; raw timings include all packets','scored_updates':len(latencies),'model':a.model,'source':source.name,'chunks':len(scores),'cadence_ms':500,'p50_roundtrip_ms':float(np.percentile(latencies,50)),'p95_roundtrip_ms':float(np.percentile(latencies,95)),'p95_update_interval_ms':float(np.percentile(intervals,95)),'median_update_interval_ms':float(np.median(intervals)),'realtime_factor_mean':float(np.mean(latencies)/500),'p95_under_500ms':bool(np.percentile(latencies,95)<500),'offline_stream_parity':True,'raw_audio_retained':False,'raw_timings_ms':all_latencies,'raw_results':scores,'scope':'Timed PCM replay of a partial fake; physical microphone permissions/hardware and cellular capture not tested.'}
    output=Path('reports/voiceguard/streaming.json');output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if not k.startswith('raw')},indent=2))
    path=Path('reports/voiceguard/evaluation.json')
    if path.exists():
        report=json.loads(path.read_text())
        for r in report['requirements']:
            if r['requirement']=='500 ms end-to-end streaming':r.update(status='PASS' if result['p95_under_500ms'] else 'FAIL',detail=f"Timed replay p95 round trip {result['p95_roundtrip_ms']:.1f} ms; median update interval {result['median_update_interval_ms']:.1f} ms; parity verified.")
        path.write_text(json.dumps(report,indent=2))
if __name__=='__main__':asyncio.run(main())

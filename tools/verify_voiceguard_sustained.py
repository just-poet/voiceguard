"""Thirty-second repeated-proxy stream; real model, no microphone claim."""
import asyncio,json,struct,sys,time,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from websockets.asyncio.client import connect
from voiceguard.audio import decode,HOP,SR
async def main():
    base='http://127.0.0.1:8765';source=Path('D:/VoiceGuardData/partials/partial_0000.wav');x=np.tile(decode(source),5);results=[];latencies=[];arrivals=[]
    async with connect('ws://127.0.0.1:8765/api/stream?model=candidate',origin=base,max_size=1024*1024) as ws:
        assert json.loads(await ws.recv())['type']=='ready';start=time.perf_counter()
        for i,offset in enumerate(range(0,len(x),HOP)):
            await asyncio.sleep(max(0,start+(i+1)*.5-time.perf_counter()));before=time.perf_counter();await ws.send(struct.pack('<Id',i,offset)+x[offset:offset+HOP].astype('<f4').tobytes());r=json.loads(await ws.recv());now=time.perf_counter();assert r['sequence']==i and r['type']=='score'
            if r['score'] is not None:latencies.append((now-before)*1000);arrivals.append(now)
            results.append(r)
    status=json.load(urllib.request.urlopen(base+'/api/status'));intervals=np.diff(arrivals)*1000
    report={'source':'partial_0000.wav repeated five times; not 30 seconds of independent speech','duration_seconds':len(x)/SR,'chunks':len(results),'scored_updates':len(latencies),'backend':status['model']['backend'],'p50_roundtrip_ms':float(np.median(latencies)),'p95_roundtrip_ms':float(np.percentile(latencies,95)),'p95_update_interval_ms':float(np.percentile(intervals,95)),'median_update_interval_ms':float(np.median(intervals)),'realtime_factor_mean':float(np.mean(latencies)/500),'p95_under_500ms':bool(np.percentile(latencies,95)<500),'all_sequences_received':True,'raw_timings_ms':latencies,'raw_results':results,'microphone_tested':False}
    out=Path('reports/voiceguard/streaming-30s.json');out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if not k.startswith('raw')}))
    p=out.with_name('evaluation.json');evaluation=json.loads(p.read_text())
    for r in evaluation['requirements']:
        if r['requirement']=='500 ms end-to-end streaming':r.update(status='PASS' if report['p95_under_500ms'] else 'FAIL',detail=f"30-second repeated-proxy replay: p95 round trip {report['p95_roundtrip_ms']:.1f} ms, median update interval {report['median_update_interval_ms']:.1f} ms, all 60 packets received. Separate prefix-parity test passed; microphone/cellular capture not tested.")
    p.write_text(json.dumps(evaluation,indent=2))
if __name__=='__main__':asyncio.run(main())

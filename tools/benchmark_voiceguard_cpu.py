"""CPU thread selection using fixed noise, not accuracy/test-set model selection."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from voiceguard.model import Engine,N_SAMPLES
e=Engine(backend='torch');x=np.random.default_rng(917).normal(0,.04,N_SAMPLES);rows=[];reference=None
for threads in (1,2,4,6,8):
    torch.set_num_threads(threads)
    for _ in range(2):e.features(x)
    times=[]
    for _ in range(8):
        start=time.perf_counter();features,score,_=e.features(x);times.append((time.perf_counter()-start)*1000)
        if reference is None:reference=(features.copy(),score)
        np.testing.assert_allclose(features,reference[0],rtol=1e-4,atol=1e-4)
        assert abs(score-reference[1])<1e-4
    row={'threads':threads,'p50_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'raw_ms':times};rows.append(row);print(json.dumps(row),flush=True)
report={'input':'Fixed seeded noise, 64600 samples; runtime tuning only, not speech evaluation','blas_threads':1,'configurations':rows,'recommended_threads':min(rows,key=lambda r:r['p95_ms'])['threads'],'score_parity':True}
Path('reports/voiceguard/cpu-tuning.json').write_text(json.dumps(report,indent=2))

import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
sys.path.insert(0,str(Path('experiments/aasist').resolve()))
import torch,numpy as np
from architecture import Model
from callguard.audio import decode,condition
from callguard.dataset import read_manifest
from callguard.metrics import eer
root=Path('experiments/aasist');torch.set_num_threads(4)
m=Model(json.loads((root/'config.json').read_text())['model_config']);m.load_state_dict(torch.load(root/'weights.pth',map_location='cpu',weights_only=True));m.eval()
rows=[r for r in read_manifest('data/indictts/manifest.jsonl') if r['split']=='dev'];out={}
with torch.inference_mode():
 for name in ['clean','mulaw_noise10']:
  scores=[];times=[]
  for r in rows:
   x=condition(decode(r['path'])[0],name,42)[:32000];t=time.perf_counter();emb,logit=m(torch.from_numpy(x.copy()).float().unsqueeze(0));times.append((time.perf_counter()-t)*1000);scores.append(float(torch.softmax(logit,dim=1)[0,0]))
  out[name]={'eer':eer([r['label'] for r in rows],scores),'p95_ms':float(np.percentile(times,95))};print(name,out[name],flush=True)
(root/'development-reference.json').write_text(json.dumps({'note':'Official pretrained reference on first 2 seconds only; no fitting or test data used. Short input differs from original training length.','results':out},indent=2))

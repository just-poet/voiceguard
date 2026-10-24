"""Check unchanged encoder/head scores on development prefixes across runtimes."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from voiceguard.model import Engine
from voiceguard.audio import decode,SR
from voiceguard.evaluate import score_features
reference=Engine(backend='torch');candidate=Engine(backend='onnx');groups={};records=[]
for manifest in [Path('data/indictts/manifest.jsonl'),Path('D:/VoiceGuardData/subsets/asvspoof2019_la/manifest.jsonl')]:
    for line in manifest.read_text().splitlines():
        r=json.loads(line);key=(r['dataset'],r.get('language'),r['label'])
        if r['split']!='dev' or groups.get(key,0)>=2:continue
        x=decode(manifest.parent/r['path'],max_seconds=60)
        if len(x)<2*SR:continue
        groups[key]=groups.get(key,0)+1
        for duration in (2,4):
            y=x[:duration*SR];a,sa,_=reference.features(y);b,sb,_=candidate.features(y)
            delta=abs(score_features(reference,a,sa,'candidate')-score_features(candidate,b,sb,'candidate'))
            assert abs(sa-sb)<1e-4 and delta<1e-3
            records.append({'development_source_id':r['source_id'],'duration':len(y)/SR,'baseline_absolute_error':abs(sa-sb),'candidate_absolute_error':delta})
result={'passed':True,'cases':len(records),'baseline_tolerance':1e-4,'candidate_tolerance':1e-3,'max_candidate_absolute_error':max(r['candidate_absolute_error'] for r in records),'records':records,'thresholds_changed':False,'weights_changed':False}
Path('reports/voiceguard/onnx-head-parity.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'}))

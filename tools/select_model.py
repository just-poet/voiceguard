"""Select a candidate using development EER only, then freeze its thresholds."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from callguard.audio import decode
from callguard.dataset import read_manifest
from callguard.detector import Detector
from callguard.metrics import eer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest');p.add_argument('candidates',nargs='+');p.add_argument('--output',default='models/detector.json')
    args=p.parse_args()
    candidates=[(Path(path),json.loads(Path(path).read_text(encoding='utf-8'))) for path in args.candidates]
    path, model=min(candidates,key=lambda pair:pair[1]['training']['development_eer_percent'])
    rows=read_manifest(args.manifest)
    dev=[r for r in rows if r['split']=='dev']
    detector=Detector(path)
    labels=[];context=[];local=[]
    for row in dev:
        result=detector.analyze(decode(row['path'])[0])
        for b in result['bins']:
            if b['score'] is not None:
                labels.append(row['label']);context.append(b['context_score']);local.append(b['score'])
    model['context_bin_threshold']=eer(labels,context)['threshold']
    model['local_threshold']=eer(labels,local)['threshold']
    model['training']['development_text_groups']=sorted({r['text_group_sha256'] for r in rows if r['split']!='test' and r.get('text_group_sha256')})
    model['training']['selection']={'criterion':'minimum clean development clip EER; no test scores used',
                                   'candidates':{p.name:m['training']['development_eer_percent'] for p,m in candidates},
                                   'selected':path.name}
    output=Path(args.output);output.parent.mkdir(exist_ok=True,parents=True)
    output.write_text(json.dumps(model,separators=(',',':'),allow_nan=False),encoding='utf-8')
    receipt={'model_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'selection':model['training']['selection'],
             'thresholds':{k:model[k] for k in ('threshold','local_threshold','context_bin_threshold')},
             'development_clips':len(dev),'test_scores_used':False}
    Path('reports').mkdir(exist_ok=True)
    Path('reports/model-selection.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()

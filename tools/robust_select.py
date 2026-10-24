"""A bounded model/view-weight comparison using development audio only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from callguard.audio import CONDITIONS,condition,decode
from callguard.dataset import read_manifest
from callguard.detector import Detector
from callguard.metrics import eer

STRATEGIES={'equal_views':[1/3,1/3,1/3],'native_anchor':[.8,.1,.1],'native_only':[1.,0.,0.]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest');p.add_argument('models',nargs='+');p.add_argument('--max-clean-eer',type=float,default=10.5106)
    args=p.parse_args()
    rows=[r for r in read_manifest(args.manifest) if r['split']=='dev']
    audio=[decode(r['path'])[0] for r in rows]
    all_results=[];caches={}
    cache_root=Path('reports/development-cache');cache_root.mkdir(parents=True,exist_ok=True)
    for path in args.models:
        path=Path(path);model_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        key=hashlib.sha256(json.dumps([model_hash,'dev-view-v1',[(r['sha256'],r['label']) for r in rows]]).encode()).hexdigest()
        cache_path=cache_root/(key+'.json')
        if cache_path.exists():cache=json.loads(cache_path.read_text(encoding='utf-8'))
        else:
            detector=Detector(path);cache={}
            for name in CONDITIONS:
                trials=[]
                for i,(row,x) in enumerate(zip(rows,audio)):
                    result=detector.analyze(condition(x,name,seed=42+i))
                    bins=[b for b in result['bins'] if b['end']>=2]
                    trials.append({'label':row['label'],'context':[[b['context_views'][v] for v in ('native','mulaw','alaw')] for b in bins],
                                   'local':[[b['views'][v] for v in ('native','mulaw','alaw')] for b in bins]})
                cache[name]=trials
                print(f'Development views: {path.name} / {name}',flush=True)
            cache_path.write_text(json.dumps(cache,separators=(',',':')),encoding='utf-8')
        caches[str(path)]=cache
        for strategy,weights in STRATEGIES.items():
            metrics={}
            for name,trials in cache.items():
                labels=[t['label'] for t in trials]
                scores=[float(np.mean(sorted(np.asarray(t['context'])@weights)[-2:])) for t in trials]
                metrics[name]=eer(labels,scores)['eer_percent']
            all_results.append({'model':str(path),'strategy':strategy,'view_weights':weights,'conditions':metrics,
                                'worst_eer_percent':max(metrics.values()),'clean_eer_percent':metrics['clean'],
                                'eligible':metrics['clean']<=args.max_clean_eer})
    eligible=[r for r in all_results if r['eligible']]
    if not eligible:raise ValueError('No candidate satisfies the predeclared development clean-EER guardrail.')
    best=min(eligible,key=lambda r:(r['worst_eer_percent'],r['clean_eer_percent']))
    model=json.loads(Path(best['model']).read_text(encoding='utf-8'))
    model['view_weights']=dict(zip(('native','mulaw','alaw'),best['view_weights']))
    trials=caches[best['model']]['clean'];weights=best['view_weights']
    labels=[t['label'] for t in trials]
    scores=[float(np.mean(sorted(np.asarray(t['context'])@weights)[-2:])) for t in trials]
    local_scores=[];context_scores=[];bin_labels=[]
    for t in trials:
        local_scores.extend((np.asarray(t['local'])@weights).tolist());context_scores.extend((np.asarray(t['context'])@weights).tolist())
        bin_labels.extend([t['label']]*len(t['local']))
    model['threshold']=eer(labels,scores)['threshold']
    model['local_threshold']=eer(bin_labels,local_scores)['threshold']
    model['context_bin_threshold']=eer(bin_labels,context_scores)['threshold']
    receipt={'objective':'minimize worst absolute development EER, then clean EER, among candidates satisfying the clean-EER guardrail',
             'clean_guardrail_percent':args.max_clean_eer,'guardrail_basis':'initial selected development clean EER 8.5106% plus 2 percentage points',
             'test_scores_used_for_selection':False,'development_clips':len(rows),'candidates':all_results,'selected':best,
             'note':'This round followed an initial test robustness failure. All changes in this round were selected using development data; the initial test report is retained.'}
    model['training']['development_eer_percent']=best['clean_eer_percent']
    model['training']['selection']=receipt
    all_rows=read_manifest(args.manifest)
    model['training']['development_text_groups']=sorted({r['text_group_sha256'] for r in all_rows if r['split']!='test' and r.get('text_group_sha256')})
    output=Path('models/detector.json');output.write_text(json.dumps(model,separators=(',',':'),allow_nan=False),encoding='utf-8')
    receipt['model_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
    Path('reports/robust-selection.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()

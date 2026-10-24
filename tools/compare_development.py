"""Compare fixed candidate models on development channels; never reads test audio."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from callguard.audio import CONDITIONS,condition,decode
from callguard.dataset import read_manifest
from callguard.detector import Detector
from callguard.metrics import eer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest');p.add_argument('models',nargs='+');p.add_argument('--output',default='reports/development-comparison.json')
    args=p.parse_args()
    rows=[r for r in read_manifest(args.manifest) if r['split']=='dev']
    audio=[decode(r['path'])[0] for r in rows];labels=[r['label'] for r in rows]
    report={'selection_objective':'minimize worst absolute development EER across all declared conditions; clean EER breaks ties',
            'development_clips':len(rows),'test_audio_used':False,'models':{}}
    for path in args.models:
        detector=Detector(path);metrics={}
        for name in CONDITIONS:
            scores=[detector.analyze(condition(x,name,seed=42+i))['summary']['score'] for i,x in enumerate(audio)]
            metrics[name]=eer(labels,scores)['eer_percent']
            print(f'{Path(path).name} {name}: {metrics[name]:.2f}% EER',flush=True)
        report['models'][Path(path).name]={'conditions':metrics,'worst_eer_percent':max(metrics.values()),'clean_eer_percent':metrics['clean']}
        Path(args.output).parent.mkdir(exist_ok=True,parents=True)
        Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

"""Additional real-container codec conditions; model and dev thresholds remain frozen."""
import json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import soundfile as sf
from voiceguard.audio import reference_codec,decode,quality,SR
from voiceguard.model import Engine
from voiceguard.datasets import load_manifest
from voiceguard.metrics import metrics
ROOT=Path(__file__).resolve().parents[1];DATA=Path(os.environ.get('VOICEGUARD_DATA','D:/VoiceGuardData'));OUT=ROOT/'reports/voiceguard'
report=json.loads((OUT/'evaluation.json').read_text());raw=[json.loads(s) for s in (OUT/'raw-scores.jsonl').read_text().splitlines()]
conditions=('g711_mulaw_wav','g711_alaw_wav','g711_mulaw_wav_noise10');raw=[r for r in raw if r['condition'] not in conditions]
selected={(r['dataset'],r['source_id']) for r in raw if r['condition']=='clean' and r['duration_condition'] is None}
rows=[]
for manifest in [ROOT/'data/indictts/manifest.jsonl',DATA/'subsets/asvspoof2019_la/manifest.jsonl',DATA/'subsets/in_the_wild/manifest.jsonl']:
    rows.extend(r for r in load_manifest(manifest) if r['split']=='test' and (r['dataset'],r['source_id']) in selected)
engine=Engine();forbidden=set(engine.training['train_hashes'])|set(engine.training['development_hashes']);assert not any(r['sha256'] in forbidden for r in rows)
from voiceguard.evaluate import score_features
started=time.time()
for i,r in enumerate(rows):
    x=decode(r['path'],max_seconds=60)[:10*SR]
    for condition in conditions:
        y=reference_codec(x,condition,42);start=time.perf_counter();features,base,_=engine.features(y);ms=(time.perf_counter()-start)*1000;q=quality(y)
        for model in ('baseline','candidate'):
            score=score_features(engine,features,base,model);cal=engine.calibration[model]
            raw.append({'dataset':r['dataset'],'source_id':r['source_id'],'sha256':r['sha256'],'label':r['label'],'generator':r.get('generator'),'model':model,'condition':condition,'duration_condition':None,'segment':None,'score':score,'accepted':q['usable'] and abs(score-cal['threshold'])>=cal['margin'],'inference_ms':ms,'actual_seconds':len(y)/SR})
    if i%25==0:print(f'Reference codecs {i}/{len(rows)} elapsed {time.time()-started:.0f}s',flush=True)
for dataset,models in report['groups'].items():
    for model,groups in models.items():
        for condition in conditions:
            trials=[r for r in raw if r['dataset']==dataset and r['model']==model and r['condition']==condition]
            groups[condition]=metrics([r['label'] for r in trials],[r['score'] for r in trials],engine.calibration[model]['threshold'],[r['accepted'] for r in trials],[r['inference_ms'] for r in trials])
        clean=groups['clean']['eer_percent'];worst=max(v['eer_percent'] for k,v in groups.items() if k!='clean' and v['eer_percent'] is not None)
        report['robustness'][dataset][model].update(worst_eer_percent=worst,ratio=worst/clean if clean else None,within_2_5x=worst<=2.5*clean)
for requirement in report['requirements']:
    if requirement['requirement'].startswith('Telephone EER'):
        requirement.update(status='PASS' if all(v['candidate']['within_2_5x'] for v in report['robustness'].values()) else 'FAIL',detail='Includes actual libsndfile ULAW/ALAW WAV encode/decode conditions and combined phone/noise; identical held-out recordings.')
report['configuration']['transforms'].update({'g711_mulaw_wav':'16k->8k, 300-3400 Hz, actual ULAW WAV encode/decode via libsndfile, ->16k','g711_alaw_wav':'Same channel, actual ALAW WAV encode/decode','g711_mulaw_wav_noise10':'Actual ULAW WAV phone channel + 10 dB Gaussian noise, seed 42'})
report['reference_codec_run']={'libsndfile':sf.__libsndfile_version__,'clips':len(rows),'conditions':conditions,'elapsed_seconds':time.time()-started,'note':'Original manual-codeword conditions remain separately labelled; small quantizer-boundary differences versus FFmpeg are documented. No retraining or threshold changes.'}
(OUT/'raw-scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in raw));(OUT/'evaluation.json').write_text(json.dumps(report,indent=2));print('Reference codec evaluation complete.',flush=True)

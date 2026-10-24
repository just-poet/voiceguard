"""Descriptive language/attack breakdowns; never fit or change any model threshold."""
import json,sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from voiceguard.metrics import metrics
from voiceguard.metrics import binary_metrics
from voiceguard.audio import decode,transform,quality,SR
from voiceguard.model import N_SAMPLES
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'reports/voiceguard'
report=json.loads((out/'evaluation.json').read_text());raw=[json.loads(s) for s in (out/'raw-scores.jsonl').read_text().splitlines()]
language={}
for s in (ROOT/'data/indictts/manifest.jsonl').read_text(encoding='utf-8').splitlines():
    r=json.loads(s);language[(r['dataset'],r['source_id'])]=r['language']
def aggregate(rows,model):
    return metrics([r['label'] for r in rows],[r['score'] for r in rows],report['configuration']['model']['calibration'][model]['threshold'],[r['accepted'] for r in rows],[r['inference_ms'] for r in rows])
report['language_groups']={}
for lang in ('English','Hindi','Tamil'):
    report['language_groups'][lang]={}
    for model in ('baseline','candidate'):
        rows=[r for r in raw if r['model']==model and r['condition']=='clean' and r['duration_condition'] is None and language.get((r['dataset'],r['source_id']))==lang]
        report['language_groups'][lang][model]=aggregate(rows,model) if rows else None
report['language_scope']='Held-out IndicTTS subset only; small counts, unknown speaker/tool identities, no language-wide claim.'
report['asv_attack_groups']={}
asv=[r for r in raw if r['dataset']=='ASVspoof2019-LA' and r['condition']=='clean' and r['duration_condition'] is None]
for attack in sorted({r['generator'] for r in asv if r['label']}):
    report['asv_attack_groups'][attack]={}
    for model in ('baseline','candidate'):
        rows=[r for r in asv if r['model']==model and (not r['label'] or r['generator']==attack)]
        report['asv_attack_groups'][attack][model]=aggregate(rows,model)
report['asv_attack_scope']='Each evaluation attack versus the same bona-fide evaluation controls. Attack IDs follow the official protocol; unknown IndicTTS generator identities prevent a universal unseen-generator claim.'
localization=[json.loads(s) for s in (out/'localization-scores.jsonl').read_text().splitlines()]
partial_manifest=Path(os.environ.get('VOICEGUARD_DATA','D:/VoiceGuardData'))/'partials/manifest.jsonl'
quality_lookup={}
for line in partial_manifest.read_text().splitlines():
    row=json.loads(line);x=decode(partial_manifest.parent/row['path'])
    for condition in ('clean','telephone_noise10'):
        y=transform(x,condition,42)
        for start in range(0,len(y),8000):
            end=min(start+8000,len(y));quality_lookup[(row['source_id'],condition,start/SR,end/SR)]=quality(y[max(0,end-N_SAMPLES):end])['usable']
for r in localization:
    cal=report['configuration']['model']['calibration'][r['model']]
    usable=quality_lookup.get((r['source_id'],r['condition'],r['start'],r['end']),True)
    thresh=0.50 if r['model']=='candidate' else cal['threshold']+cal['margin']
    r['quality_usable']=usable;r['ui_prediction']=bool(r['score'] is not None and usable and r['score']>=thresh)
report['localization_ui']={}
for model in ('baseline','candidate'):
    for condition in ('clean','telephone_noise10'):
        for control in ('partial','sham_real_to_real','full_real','full_synthetic'):
            rows=[r for r in localization if r['model']==model and r['condition']==condition and r['control_type']==control]
            report['localization_ui'][model+'/'+condition+'/'+control]=binary_metrics([r['label'] for r in rows],[r['ui_prediction'] for r in rows])
report['localization_ui_note']='Displayed Likely synthetic alerts require quality acceptance and score >= development threshold + abstention margin. Inconclusive/warm-up bins count as no alert. Raw-threshold F1 is separately retained in localization/localization_controls.'
(out/'localization-scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in localization))
(out/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
dev=json.loads((out/'development-scores.json').read_text());polarity={}
for dataset in sorted({r['dataset'] for r in dev}):
    polarity[dataset]={}
    for model in ('baseline','candidate'):
        rows=[r for r in dev if r['dataset']==dataset];m=metrics([r['label'] for r in rows],[r[model+'_score'] for r in rows],report['configuration']['model']['calibration'][model]['threshold'])
        polarity[dataset][model]={'development_auc_higher_is_synthetic':m['roc_auc'],'count':m['count'],'direction_changed':False}
(out/'polarity-check.json').write_text(json.dumps(polarity,indent=2));print(json.dumps(report['language_groups'],indent=2))

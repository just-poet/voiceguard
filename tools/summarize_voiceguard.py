"""Generate review evidence from completed artifacts, never from expected targets."""
import collections,csv,html,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DATA=Path(os.environ.get('VOICEGUARD_DATA','D:/VoiceGuardData'));OUT=ROOT/'reports/voiceguard'

def data_status():
    status={'recorded_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'root':str(DATA),'archives':{},'asv_extracted':{},'subsets':{}}
    for name in ('asv_train_parquet','asv_dev_parquet','asv_test_parquet','in_the_wild'):
        receipt=DATA/'archives'/f'{name}.provenance.json';progress=DATA/'archives'/f'{name}.progress.json'
        status['archives'][name]=json.loads(receipt.read_text()) if receipt.exists() else {'complete_archive':False,'progress':json.loads(progress.read_text()) if progress.exists() else None}
    for split in ('train','dev','test'):
        p=DATA/'datasets/asvspoof2019_la_mirror'/f'{split}-VERIFIED.json'
        status['asv_extracted'][split]=json.loads(p.read_text()) if p.exists() else {'complete_official_split_ids':False}
    all_asv=DATA/'datasets/asvspoof2019_la_mirror/ALL-FILES-VERIFIED.json'
    status['asv_all_files']=json.loads(all_asv.read_text()) if all_asv.exists() else {'all_manifest_sha256_checked':False}
    itw_receipt=DATA/'datasets/in_the_wild/VERIFIED.json'
    status['in_the_wild_extracted']=json.loads(itw_receipt.read_text()) if itw_receipt.exists() else {'verified':False}
    for name,path in [('ASVspoof2019-LA',DATA/'subsets/asvspoof2019_la/manifest.jsonl'),('In-the-Wild',DATA/'subsets/in_the_wild/manifest.jsonl'),('IndicTTS',ROOT/'data/indictts/manifest.jsonl'),('Partial proxy',DATA/'partials/manifest.jsonl')]:
        if path.exists():
            rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
            status['subsets'][name]={'records':len(rows),'split_counts':dict(collections.Counter(r['split'] for r in rows)),'manifest':str(path)}
    (OUT/'data-status.json').write_text(json.dumps(status,indent=2),encoding='utf-8');return status

def main():
    OUT.mkdir(parents=True,exist_ok=True);status=data_status();p=OUT/'evaluation.json'
    if not p.exists():print('Data status written; evaluation not yet complete.');return
    report=json.loads(p.read_text());training=json.loads((OUT/'training.json').read_text())['provenance'];stream_path=OUT/'streaming-30s.json';stream=json.loads(stream_path.read_text()) if stream_path.exists() else None
    esc=html.escape;rows=[]
    for dataset,models in report['groups'].items():
        for model,conditions in models.items():
            for condition,m in conditions.items():rows.append(f'<tr><td>{esc(dataset)}</td><td>{model}</td><td>{condition}</td><td>{m["count"]}</td><td>{m["eer_percent"]}</td><td>{m["roc_auc"]:.4f}</td><td>{m["accepted_fraction"]:.1%}</td></tr>')
    ledger=''.join(f'<tr><td>{esc(r["requirement"])}</td><td>{r["status"]}</td><td>{esc(r["detail"])}</td></tr>' for r in report['requirements'])
    content=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>VoiceGuard measured results</title><style>body{{font:16px system-ui;color:#163d37;max-width:1200px;margin:40px auto;padding:24px}}h1{{font-size:42px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #d9e1de}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f3;padding:20px}}.note{{padding:20px;background:#fff1d5}}</style><h1>VoiceGuard: measured evidence</h1><p>Free local CPU application · AASIST-L + trained adaptation · English / Hindi / Tamil exploratory coverage</p><p class="note">{esc(report['note'])} Acoustic authenticity does not establish caller identity or scam intent.</p><h2>Training</h2><p>{esc(training['method'])} Selected using development data only. {training['train_windows']} clean/augmented training windows; {training['development_clips']} development clips; {training['epochs']} epochs executed.</p><h2>Requirement ledger</h2><table><tr><th>Requirement</th><th>Status</th><th>Evidence</th></tr>{ledger}</table><h2>Held-out results</h2><table><tr><th>Dataset</th><th>Model</th><th>Condition</th><th>n</th><th>EER %</th><th>ROC-AUC</th><th>Accepted</th></tr>{''.join(rows)}</table><h2>500 ms replay timing</h2><pre>{esc(json.dumps({k:v for k,v in stream.items() if not k.startswith('raw')} if stream else {'status':'NOT RUN'},indent=2))}</pre><h2>Partial-fake window baseline</h2><p>{esc(report.get('localization_note','Not evaluated'))}</p><pre>{esc(json.dumps({'raw_threshold_partial_only':{k:v for k,v in report['localization_controls'].items() if k.endswith('/partial')},'displayed_alert_partial_only':{k:v for k,v in report.get('localization_ui',{}).items() if k.endswith('/partial')}},indent=2))}</pre><h2>Data acquisition receipts</h2><pre>{esc(json.dumps(status,indent=2))}</pre><p>Full source, raw scores, configurations, model card, architecture, requirement matrix, demonstration script and licence notes accompany this report. Docker and physical microphone hardware tests remain NOT RUN unless separately evidenced.</p></html>'''
    (OUT/'submission.html').write_text(content,encoding='utf-8');print('Wrote data-status.json and submission.html')
if __name__=='__main__':main()

"""Protocol adapters and source-lineage checks."""
import csv,hashlib,json
from pathlib import Path

def load_manifest(path):
    path=Path(path).resolve();rows=[]
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():continue
        r=json.loads(line);r['path']=str((path.parent/r['path']).resolve())
        if r.get('split') not in ('train','dev','test') or r.get('label') not in (0,1):raise ValueError('Invalid split or label')
        actual=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()
        if r.get('sha256') and r['sha256']!=actual:raise ValueError('Manifest file checksum changed')
        r['sha256']=actual;r.setdefault('source_id',actual);r.setdefault('dataset','IndicTTS exploratory');r.setdefault('speaker','unknown');r.setdefault('generator','unknown')
        intervals=r.get('fake_segments',[]);previous=0
        for start,end in intervals:
            if not 0<=previous<=start<end:raise ValueError('Invalid segment intervals')
            previous=end
        rows.append(r)
    validate_lineage(rows);return rows

def validate_lineage(rows):
    seen={};speakers={};text={}
    for r in rows:
        for source in [r['sha256'],*r.get('source_hashes',[])]:
            if source in seen and seen[source]!=r['split']:raise ValueError('Parent-source leakage across splits')
            seen[source]=r['split']
        if r.get('speaker','unknown')!='unknown':
            key=(r.get('dataset'),r['speaker'])
            if key in speakers and speakers[key]!=r['split']:raise ValueError('Speaker leakage across splits')
            speakers[key]=r['split']
        key=r.get('text_group_sha256')
        if key:
            if key in text and text[key]!=r['split']:raise ValueError('Transcript leakage across splits')
            text[key]=r['split']

def asvspoof_adapter(root,output):
    root=Path(root);rows=[]
    for source_split,split in [('train','train'),('dev','dev'),('eval','test')]:
        matches=list(root.rglob(f'*.cm.{source_split}.*.txt'))
        if len(matches)!=1:raise ValueError(f'Expected exactly one {source_split} protocol')
        audio={p.stem:p for p in root.rglob('*.flac') if f'LA_{source_split}' in str(p)}
        for line in matches[0].read_text().splitlines():
            speaker,key,_,attack,label=line.split()
            if key not in audio:raise ValueError(f'Missing official audio: {key}')
            rows.append({'path':str(audio[key].resolve()),'label':int(label=='spoof'),'speaker':speaker,'generator':attack if label=='spoof' else 'human','source_id':key,'dataset':'ASVspoof2019-LA','language':'English','split':split,'official_split':source_split})
    Path(output).write_text(''.join(json.dumps(r)+'\n' for r in rows));return len(rows)

def itw_adapter(root,output):
    root=Path(root);meta=list(root.rglob('meta.csv'))
    if len(meta)!=1:raise ValueError('Expected one meta.csv')
    rows=[]
    for r in csv.DictReader(meta[0].open(encoding='utf-8-sig')):
        rows.append({'path':str((meta[0].parent/r['file']).resolve()),'label':int(r['label']=='spoof'),'speaker':r['speaker'],'generator':'unknown' if r['label']=='spoof' else 'human','source_id':r['file'],'dataset':'In-the-Wild','language':'English','split':'test','official_split':'evaluation-only'})
    Path(output).write_text(''.join(json.dumps(r)+'\n' for r in rows));return len(rows)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['asvspoof','itw']);p.add_argument('root');p.add_argument('output');a=p.parse_args();print((asvspoof_adapter if a.dataset=='asvspoof' else itw_adapter)(a.root,a.output))

"""Bounded ASVspoof mirror subset, matched to official protocol IDs and labels."""
import hashlib,json,time,urllib.request,urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from prepare_benchmark_subsets import balanced
DATASET='Bisher/ASVspoof_2019_LA';REV='aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c';root=Path('D:/VoiceGuardData/subsets/asvspoof2019_la');root.mkdir(parents=True,exist_ok=True)
def get(url,json_data=False):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url,timeout=45) as response:data=response.read()
            return json.loads(data) if json_data else data
        except Exception:
            if attempt==4:raise
            time.sleep(60 if getattr(__import__('sys').exc_info()[1],'code',None)==429 else attempt+1)
records=[]
for source_split,hf_split,split,n in [('train','train','train',400),('dev','validation','dev',80),('eval','test','test',80)]:
    path=root/f'ASVspoof2019.LA.cm.{source_split}.{"trn" if split=="train" else "trl"}.txt'
    if not path.exists():
        protocol_base='https://huggingface.co/datasets/Nemez1z/asvspoof-2019-la/resolve/4ad1b061fa6dc502b30888ee467ed4115fdb44eb/ASVspoof2019_LA_cm_protocols/'
        path.write_bytes(get(protocol_base+path.name))
    rows=[]
    for index,line in enumerate(path.read_text().splitlines()):
        speaker,key,_,attack,label=line.split();rows.append({'row_index':index,'source_id':key,'speaker':speaker,'generator':attack if label=='spoof' else 'human','label':int(label=='spoof'),'split':split,'language':'English','dataset':'ASVspoof2019-LA','official_split':source_split})
    # Bound metadata requests: deterministic pages spanning each class/tool group.
    groups={}
    for r in rows:groups.setdefault((r['label'],r['generator']),[]).append(r['row_index'])
    pages=set()
    for (label,tool),indices in groups.items():
        count=5 if label==0 and split=='train' else 2
        for j in range(count):pages.add(indices[int(j*(len(indices)-1)/max(count-1,1))]//100*100)
    pages=sorted(pages);pool_rows=[r for r in rows if r['row_index']//100*100 in pages];selected=balanced(pool_rows,n,42)
    def page(offset):
        query=urllib.parse.urlencode({'dataset':DATASET,'config':'default','split':hf_split,'offset':offset,'length':100})
        response=get('https://datasets-server.huggingface.co/rows?'+query,True)
        return {r['row_idx']:r['row'] for r in response['rows']}
    lookup={}
    with ThreadPoolExecutor(max_workers=1) as pool:
        for batch in pool.map(page,pages):lookup.update(batch)
    def download(row):
        item=lookup[row['row_index']]
        if item['audio_file_name']!=row['source_id'] or item['key']!=row['label'] or item['speaker_id']!=row['speaker']:raise ValueError('Mirror differs from official protocol')
        url=item['audio'][0]['src']
        if REV not in url:raise ValueError('Unexpected mirror revision')
        path=root/'audio'/f'{row["source_id"]}.flac';path.parent.mkdir(exist_ok=True)
        if not path.exists():path.write_bytes(get(url))
        return {**row,'path':'audio/'+path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'mirror_dataset':DATASET,'mirror_revision':REV,'format_note':'Dataset viewer FLAC; may be re-encoded, official IDs and labels checked'}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for row in pool.map(download,selected):
            records.append(row)
            if len(records)%50==0:print(f'ASVspoof verified-ID subset {len(records)}/1120',flush=True)
(root/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
(root/'PROVENANCE.json').write_text(json.dumps({'dataset':'ASVspoof2019-LA','official_source':'https://datashare.ed.ac.uk/handle/10283/3336','mirror_dataset':DATASET,'revision':REV,'protocol_crosscheck':'Each ID, speaker and label matched to protocol; original split preserved','complete_benchmark':False,'clips':len(records),'seed':42,'actual_split_counts':{split:sum(r['split']==split for r in records) for split in ('train','dev','test')},'scope':'Bounded page selection; actual counts may be below requested counts','format_note':'Viewer FLAC may be re-encoded; full official archive download is separate'},indent=2))
print('ASVspoof subset ready',flush=True)

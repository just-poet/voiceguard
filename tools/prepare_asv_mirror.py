"""Pinned mirror of original FLACs; retain official split and verify CRC when available."""
import json,time,urllib.request,zlib,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from prepare_benchmark_subsets import balanced

BASE='https://huggingface.co/datasets/Nemez1z/asvspoof-2019-la/resolve/4ad1b061fa6dc502b30888ee467ed4115fdb44eb/'
root=Path('D:/VoiceGuardData/subsets/asvspoof2019_la');root.mkdir(parents=True,exist_ok=True)
def get(name):
    for attempt in range(5):
        try:return urllib.request.urlopen(BASE+name,timeout=45).read()
        except Exception:
            if attempt==4:raise
            time.sleep(2+attempt)
rows=[]
for source_split,split,n in [('train','train',400),('dev','dev',80),('eval','test',80)]:
    protocol=f'ASVspoof2019.LA.cm.{source_split}.{"trn" if split=="train" else "trl"}.txt'
    content=get('ASVspoof2019_LA_cm_protocols/'+protocol);(root/protocol).write_bytes(content);items=[]
    for line in content.decode().splitlines():
        speaker,key,_,attack,label=line.split();items.append({'source_id':key,'speaker':speaker,'generator':attack if label=='spoof' else 'human','label':int(label=='spoof'),'split':split,'language':'English','dataset':'ASVspoof2019-LA','official_split':source_split})
    rows.extend(balanced(items,n,42))
def fetch(r):
    member=f'ASVspoof2019_LA_{r["official_split"]}/flac/{r["source_id"]}.flac';path=root/'audio'/f'{r["source_id"]}.flac';path.parent.mkdir(exist_ok=True)
    if not path.exists():path.write_bytes(get(member))
    data=path.read_bytes();return {**r,'path':'audio/'+path.name,'sha256':hashlib.sha256(data).hexdigest(),'crc32':zlib.crc32(data)&0xffffffff,'source_member':'LA/'+member,'mirror_url':BASE+member}
records=[]
with ThreadPoolExecutor(max_workers=6) as pool:
    for r in pool.map(fetch,rows):
        records.append(r)
        if len(records)%50==0:print(f'ASVspoof mirror: {len(records)}/{len(rows)}',flush=True)
(root/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
(root/'PROVENANCE.json').write_text(json.dumps({'dataset':'ASVspoof2019-LA','official_source':'https://datashare.ed.ac.uk/handle/10283/3336','mirror':BASE,'clips':len(records),'seed':42,'scope':'800 train / 160 development / 160 evaluation; original partitions preserved','official_archive_crc_verification':'pending until central directory download completes'},indent=2))
print('Mirror subset ready',flush=True)

"""Fetch deterministic official-split subsets while full archives download."""
import argparse,csv,hashlib,io,json,random,sys,zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from download_benchmarks import SOURCES,fetch_range

class RemoteZip(io.RawIOBase):
    def __init__(self,url,size):self.url=url;self.size=size;self.pos=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,n,whence=0):
        self.pos=n if whence==0 else self.pos+n if whence==1 else self.size+n
        return self.pos
    def read(self,n=-1):
        n=min(n if n>=0 else self.size-self.pos,self.size-self.pos)
        if n<=0:return b''
        print(f'ZIP index range: {self.pos} + {n}',flush=True)
        if n>1024*1024:
            starts=list(range(self.pos,self.pos+n,1024*1024));end=self.pos+n-1
            with ThreadPoolExecutor(max_workers=8) as pool:result=b''.join(pool.map(lambda start:fetch_range(self.url,start,min(start+1024*1024-1,end)),starts))
        else:result=fetch_range(self.url,self.pos,self.pos+n-1)
        self.pos+=n;return result

def read_entry(info,url):
    # Read one independent member with its local header, avoiding shared seek state.
    import struct,zlib
    start=info.header_offset;head=fetch_range(url,start,start+29)
    fields=struct.unpack('<4s5H3I2H',head);name_len,extra_len=fields[-2:]
    offset=start+30+name_len+extra_len
    data=fetch_range(url,offset,offset+info.compress_size-1)
    data=zlib.decompress(data,-15) if info.compress_type==zipfile.ZIP_DEFLATED else data
    if len(data)!=info.file_size or (zlib.crc32(data)&0xffffffff)!=info.CRC:raise ValueError('ZIP member integrity failed')
    return data

def balanced(rows,per_class,seed):
    rng=random.Random(seed);selected=[]
    for label in (0,1):
        groups={}
        for r in rows:
            if r['label']==label:groups.setdefault(r.get('generator','unknown'),[]).append(r)
        for group in groups.values():rng.shuffle(group)
        while len([r for r in selected if r['label']==label])<per_class and any(groups.values()):
            for key in sorted(groups):
                if groups[key] and len([r for r in selected if r['label']==label])<per_class:selected.append(groups[key].pop())
    return selected

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=list(SOURCES),required=True);p.add_argument('--root',default='D:/VoiceGuardData');p.add_argument('--index-only',action='store_true');a=p.parse_args()
    root=Path(a.root)/'subsets'/a.dataset;root.mkdir(parents=True,exist_ok=True)
    url,size=SOURCES[a.dataset];z=zipfile.ZipFile(RemoteZip(url,size));infos={i.filename:i for i in z.infolist()}
    (root/'archive-index.json').write_text(json.dumps({name:{'crc32':info.CRC,'bytes':info.file_size} for name,info in infos.items()}))
    if a.index_only:return
    selected=[]
    if a.dataset=='asvspoof2019_la':
        for source_split,split,n in [('train','train',400),('dev','dev',80),('eval','test',80)]:
            name=next(k for k in infos if f'.cm.{source_split}.' in k and k.endswith('.txt'))
            content=read_entry(infos[name],url).decode();(root/Path(name).name).write_text(content)
            rows=[]
            for line in content.splitlines():
                speaker,key,_,attack,label=line.split()
                audio=next((v for k,v in infos.items() if k.endswith(f'/flac/{key}.flac')),None) if False else None
                rows.append({'source_id':key,'speaker':speaker,'generator':attack if label=='spoof' else 'human','label':int(label=='spoof'),'split':split,'language':'English','dataset':'ASVspoof2019-LA','official_split':source_split})
            selected.extend(balanced(rows,n,42))
        file_map={Path(k).stem:k for k in infos if k.endswith('.flac')}
    else:
        name=next(k for k in infos if k.endswith('meta.csv'))
        content=read_entry(infos[name],url).decode('utf-8-sig');(root/'meta.csv').write_text(content,encoding='utf-8')
        rows=[]
        for r in csv.DictReader(io.StringIO(content)):
            rows.append({'source_id':r['file'],'speaker':r['speaker'],'generator':'unknown' if r['label']=='spoof' else 'human','label':int(r['label']=='spoof'),'split':'test','language':'English','dataset':'In-the-Wild','official_split':'evaluation-only'})
        selected=balanced(rows,100,42);file_map={Path(k).name:k for k in infos if k.endswith('.wav')}
    def fetch(row):
        name=file_map[row['source_id']];info=infos[name];target=root/'audio'/Path(name).name;target.parent.mkdir(exist_ok=True)
        if not target.exists():target.write_bytes(read_entry(info,url))
        return {**row,'path':'audio/'+target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'source_url':url,'source_member':name}
    records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(fetch,selected):
            records.append(r)
            if len(records)%50==0:print(f'{a.dataset}: {len(records)}/{len(selected)} clips',flush=True)
    (root/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    (root/'PROVENANCE.json').write_text(json.dumps({'dataset':a.dataset,'archive':url,'selection_seed':42,'complete_benchmark':False,'clips':len(records),'split_policy':'Official ASVspoof splits; In-the-Wild evaluation-only. No test data used for training.'},indent=2))
    print(str(root/'manifest.jsonl'),flush=True)
if __name__=='__main__':main()

"""Resumable official benchmark downloads to a configurable data drive."""
import argparse, concurrent.futures, hashlib, json, os, time, urllib.request, zipfile
from pathlib import Path

SOURCES = {
 'asvspoof2019_la': ('https://datashare.ed.ac.uk/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/download',7640952520),
 'in_the_wild': ('https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/eee168f92c367f8c82ff2cf42b6f61e362fd6211/release_in_the_wild.zip',8161489023),
}
MIRROR_BASE='https://huggingface.co/datasets/Bisher/ASVspoof_2019_LA/resolve/aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c/data/'
SOURCES.update({'asv_train_parquet':(MIRROR_BASE+'train-00000-of-00001.parquet',1586086073),'asv_dev_parquet':(MIRROR_BASE+'validation-00000-of-00001.parquet',1575535827),'asv_test_parquet':(MIRROR_BASE+'test-00000-of-00001.parquet',4381501164)})
MIRROR_SHA={'asv_train_parquet':'b4eea1063bbcfa0c1cef1b69a96ad8b787c32f662005562b899cd4b461739619','asv_dev_parquet':'9d2c340bb7f04c2d63ac018b224ecdb8c1855c789247232608b650f86c189b16','asv_test_parquet':'8159935a94426bc36308278b23cecd8c4ca2a15b778a1087a4a0441d079c86af'}
MIRROR_SHA['in_the_wild']='46665f30a6758b45642c659c7e0e80da2ba3d121b064716824a56a010ddf9e1a'

def fetch_range(url,start,end):
    request=urllib.request.Request(url+('&' if '?' in url else '?')+f'vg_range={start}-{end}',headers={'Range':f'bytes={start}-{end}','User-Agent':'VoiceGuard-research/0.2'})
    with urllib.request.urlopen(request,timeout=90) as response:
        if response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
            raise ValueError('Server did not honor exact byte range')
        content=response.read(end-start+2)
        if len(content)!=end-start+1: raise ValueError('Truncated or oversized range')
        return content

def download(name,root,workers=4):
    url,size=SOURCES[name];parts=root/'archives'/f'{name}.parts';parts.mkdir(parents=True,exist_ok=True)
    archive=root/'archives'/f'{name}{".parquet" if name.endswith("parquet") else ".zip"}';block=8*1024*1024;total=(size+block-1)//block
    if archive.exists() and archive.stat().st_size==size:
        with archive.open('rb') as existing:
            digest=hashlib.file_digest(existing,'sha256').hexdigest()
        if name in MIRROR_SHA and digest!=MIRROR_SHA[name]:
            raise ValueError(f'{name}: existing archive fails pinned SHA256 verification')
        print(f'{name}: archive already present; SHA256 {digest}',flush=True);return archive
    def one(i):
        start=i*block;end=min(start+block,size)-1;p=parts/f'{i:05d}.bin'
        if p.exists() and p.stat().st_size==end-start+1:return i
        for attempt in range(8):
            try:
                data=fetch_range(url,start,end);temp=p.with_suffix('.partial');temp.write_bytes(data);temp.replace(p);return i
            except Exception:
                if attempt==7:raise
                time.sleep(min(30,2**attempt))
    done=0;started=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(one,range(total)):
            done+=1
            if done%10==0 or done==total:
                progress={'dataset':name,'parts':done,'total':total,'percent':round(done/total*100,1),'elapsed_seconds':round(time.time()-started)}
                (root/'archives'/f'{name}.progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
    h=hashlib.sha256();temp=archive.with_suffix('.zip.partial')
    with temp.open('wb') as out:
        for i in range(total):
            data=(parts/f'{i:05d}.bin').read_bytes();h.update(data);out.write(data)
    if name in MIRROR_SHA and h.hexdigest()!=MIRROR_SHA[name]:raise ValueError('Mirror SHA256 differs from pinned Hugging Face LFS object')
    temp.replace(archive)
    (archive.with_suffix('.provenance.json')).write_text(json.dumps({'source':url,'bytes':size,'sha256':h.hexdigest(),'downloaded_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'complete_archive':True},indent=2))
    return archive

def extract(archive,root,name):
    target=(root/'datasets'/name).resolve();target.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for i,info in enumerate(z.infolist()):
            path=(target/info.filename).resolve()
            if not path.is_relative_to(target):raise ValueError('Unsafe archive path')
            if not path.exists() or (not info.is_dir() and path.stat().st_size!=info.file_size):z.extract(info,target)
            if i%10000==0:print(f'{name}: extracted {i}/{len(z.infolist())}',flush=True)
    print(f'{name}: extracted to {target}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=os.environ.get('VOICEGUARD_DATA','D:/VoiceGuardData'));p.add_argument('--dataset',choices=[*SOURCES,'all'],default='all');p.add_argument('--workers',type=int,default=4);p.add_argument('--extract',action='store_true');a=p.parse_args()
    for name in ('asvspoof2019_la','in_the_wild') if a.dataset=='all' else [a.dataset]:
        archive=download(name,Path(a.root),a.workers)
        if a.extract:extract(archive,Path(a.root),name)

"""Download a bounded, balanced public IndicTTS subset, not 18 GB of shards.

This is an exploratory text-grouped split: the dataset lacks speaker/tool IDs.
Uses the documented HF Dataset Viewer /rows API. No account or token needed.
"""
import argparse
import concurrent.futures
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import time
import unicodedata
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
from callguard.audio import SR, decode

DATASET = 'SherryT997/IndicTTS-Deepfake-Challenge-Data'
SOURCE = 'https://huggingface.co/datasets/' + DATASET
API = 'https://datasets-server.huggingface.co/rows?'
# Verified language blocks in the public train split at pinned revision below.
RANGES = {'English': range(17600, 18400, 100), 'Hindi': range(21600, 22400, 100), 'Tamil': range(29400, 30200, 100)}
REVISION = '57347517658ae989597d8cef303cffb647ed2434'


def fetch(url, limit=8*1024*1024):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent':'CallGuard-research-subset/0.1'})
            with urllib.request.urlopen(req, timeout=35) as r:
                data = r.read(limit+1)
                if len(data)>limit: raise ValueError('Remote file exceeds size limit.')
                return data
        except Exception:
            if attempt==2: raise
            time.sleep(1+attempt)


def metadata(offset):
    query = {'dataset':DATASET,'config':'default','split':'train','offset':offset,'length':100}
    return json.loads(fetch(API+urllib.parse.urlencode(query)))['rows']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--per-class',type=int,default=120)
    p.add_argument('--output',default='data/indictts')
    args=p.parse_args()
    if not 10<=args.per_class<=200: p.error('--per-class must be 10–200 for this bounded fetcher')
    out=Path(args.output).resolve();(out/'audio').mkdir(parents=True,exist_ok=True)
    card=fetch(SOURCE+'/raw/'+REVISION+'/README.md').decode('utf-8')
    if 'license: cc-by-4.0' not in card: raise ValueError('Unexpected dataset license. Review the source before use.')
    (out/'SOURCE_README.md').write_text(card,encoding='utf-8')
    candidates=[]
    offsets=sorted({i for v in RANGES.values() for i in v})
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for batch in pool.map(metadata,offsets): candidates.extend(batch)
    counts={ (lang,label):0 for lang in RANGES for label in (0,1) }
    tasks=[]
    # Deterministic shuffle avoids selecting a contiguous emotion or topic block.
    rng=np.random.default_rng(42);rng.shuffle(candidates)
    for item in candidates:
        row=item['row'];key=(row['language'],row['is_tts'])
        if key not in counts or counts[key]>=args.per_class+30: continue
        counts[key]+=1;tasks.append(item)
    print('Candidate counts:',counts,flush=True)

    def download(item):
        row=item['row'];index=item['row_idx']
        name=f'{row["language"].lower()}_{index:05d}.wav';path=out/'audio'/name
        try:
            if path.exists():
                x,_=decode(path)
            else:
                url=row['audio'][0]['src']
                parsed=urllib.parse.urlparse(url)
                if parsed.scheme!='https' or parsed.hostname!='datasets-server.huggingface.co': raise ValueError('Unexpected audio host')
                if '/--/'+REVISION+'/--/' not in parsed.path: raise ValueError('Dataset revision changed; review metadata before downloading')
                x,_=decode(fetch(url),max_seconds=60)
                if len(x)<2*SR: return None
                x=x[:6*SR]
                sf.write(path,x,SR,subtype='PCM_16')
            text=unicodedata.normalize('NFKC',row['text']).strip().casefold()
            group=hashlib.sha256(text.encode('utf-8')).hexdigest()
            fraction=int(group[:8],16)/2**32
            split='train' if fraction<.7 else 'dev' if fraction<.85 else 'test'
            return {'path':'audio/'+name,'label':int(row['is_tts']),'speaker':'unknown','generator':'unknown' if row['is_tts'] else 'human',
                    'language':row['language'],'split':split,'text_group_sha256':group,
                    'dataset':DATASET,'source_revision':REVISION,'source_train_row':index,
                    'source_id':row['id'],'source_license':'CC-BY-4.0','cropped_to_seconds':len(x)/SR}
        except Exception as e:
            print(f'Skipped row {index}: {type(e).__name__}',flush=True);return None
    records=[];counts={key:0 for key in counts};seen_hashes=set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for i,row in enumerate(pool.map(download,tasks)):
            if row:
                key=row['language'],row['label']
                digest=hashlib.sha256((out/row['path']).read_bytes()).hexdigest()
                if counts[key]<args.per_class and digest not in seen_hashes:
                    records.append(row);counts[key]+=1;seen_hashes.add(digest)
            if (i+1)%30==0:print(f'Fetched {i+1}/{len(tasks)}; retained {len(records)}',flush=True)
    if any(v<10 for v in counts.values()): raise ValueError(f'Insufficient samples: {counts}')
    records.sort(key=lambda r:(r['language'],r['source_train_row']))
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=True)+'\n' for r in records),encoding='utf-8')
    split_counts={}
    for r in records:
        key=f'{r["language"]}/{r["split"]}/label{r["label"]}';split_counts[key]=split_counts.get(key,0)+1
    provenance={'dataset':SOURCE,'revision':REVISION,'license':'CC BY 4.0','attribution':'SherryT997 / IndicTTS Deepfake Detection Challenge dataset',
                'split':'SHA256 of normalized text: 70% train / 15% dev / 15% test, drawn from public labeled train split',
                'limitations':['Speaker identities and TTS tool identities are unavailable. Their holdouts cannot be verified.',
                               'This small subset is a deadline demonstration, not the official challenge split or a multilingual robustness benchmark.',
                               'Clips longer than 6 seconds are truncated; clips under 2 seconds are excluded.',
                               'Do not infer cloning identity, fraud, or language-wide reliability from this baseline.'],
                'retained_clips':len(records),'counts':split_counts}
    (out/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps(provenance,indent=2),flush=True)


if __name__=='__main__':main()

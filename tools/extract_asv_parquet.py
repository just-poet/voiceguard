"""Stream full mirrored ASVspoof Parquets to D:, verifying every protocol row."""
import hashlib,json,time,os
from pathlib import Path
import pyarrow.parquet as pq
root=Path(os.environ.get('VOICEGUARD_DATA','D:/VoiceGuardData'));protocol_root=root/'subsets/asvspoof2019_la';target=root/'datasets/asvspoof2019_la_mirror';target.mkdir(parents=True,exist_ok=True)
for key,official,split in [('train','train','train'),('dev','dev','dev'),('test','eval','test')]:
    parquet=root/'archives'/f'asv_{key}_parquet.parquet';started=time.time()
    while not parquet.exists():
        if time.time()-started>6*3600:raise TimeoutError('Full archive still unavailable')
        time.sleep(10)
    protocol=protocol_root/f'ASVspoof2019.LA.cm.{official}.{"trn" if split=="train" else "trl"}.txt'
    expected={}
    for line in protocol.read_text().splitlines():
        speaker,source,_,tool,label=line.split();expected[source]=(speaker,tool,int(label=='spoof'))
    count=0;output=target/f'{split}.jsonl';seen=set()
    with output.open('w',encoding='utf-8') as manifest:
        for batch in pq.ParquetFile(parquet).iter_batches(batch_size=128):
            for r in batch.to_pylist():
                source=r['audio_file_name'];speaker,tool,label=expected[source]
                if r['speaker_id']!=speaker or r['key']!=label:raise ValueError('Official protocol mismatch')
                if source in seen:raise ValueError('Duplicate source in full mirror')
                seen.add(source);audio=r['audio']['bytes'];path=target/split/f'{source}.flac';path.parent.mkdir(exist_ok=True)
                if not path.exists() or hashlib.sha256(path.read_bytes()).digest()!=hashlib.sha256(audio).digest():
                    temp=path.with_suffix('.partial');temp.write_bytes(audio);temp.replace(path)
                row={'path':split+'/'+path.name,'source_id':source,'speaker':speaker,'generator':tool if label else 'human','label':label,'split':split,'official_split':official,'language':'English','dataset':'ASVspoof2019-LA','sha256':hashlib.sha256(audio).hexdigest()};manifest.write(json.dumps(row)+'\n');count+=1
            if count%1024==0:print(f'Extracted {split}: {count}/{len(expected)}',flush=True)
    if seen!=set(expected):raise ValueError('Full mirror does not contain the complete official split')
    (target/f'{split}-VERIFIED.json').write_text(json.dumps({'count':count,'complete_official_split_ids':True,'source':str(parquet),'protocol':str(protocol)},indent=2));print(f'{split} complete: {count}',flush=True)
with (target/'manifest.jsonl').open('wb') as out:
    for name in ('train','dev','test'):out.write((target/f'{name}.jsonl').read_bytes())
print('Full mirrored ASVspoof corpus extracted and protocol-verified.',flush=True)

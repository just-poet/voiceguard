"""Wait for complete archives and verify extracted audio against download content."""
import csv,hashlib,json,os,time,zipfile
from pathlib import Path
DATA=Path(os.environ.get('VOICEGUARD_DATA','D:/VoiceGuardData'))
def wait(path):
    start=time.time()
    while not path.exists():
        if time.time()-start>6*3600:raise TimeoutError(str(path))
        time.sleep(10)
archive=DATA/'archives/in_the_wild.zip';wait(archive)
target=DATA/'datasets/in_the_wild';metadata=target/'release_in_the_wild/meta.csv';wait(metadata)
with zipfile.ZipFile(archive) as z:
    # Extraction may still be writing; verify every audio file against ZIP CRC.
    import zlib
    count=0
    for info in z.infolist():
        if info.is_dir():continue
        path=target/info.filename;wait(path)
        for attempt in range(120):
            data=path.read_bytes()
            if len(data)==info.file_size and zlib.crc32(data)&0xffffffff==info.CRC:break
            time.sleep(1)
        else:raise ValueError('Extracted CRC mismatch: '+str(path))
        count+=info.filename.endswith('.wav')
rows=list(csv.DictReader(metadata.open(encoding='utf-8-sig')));assert len(rows)==count==31779
records=[]
for r in rows:
    path=metadata.parent/r['file'];assert path.is_file()
    records.append({'path':str(path.relative_to(target)),'source_id':r['file'],'label':int(r['label']=='spoof'),'speaker':r['speaker'],'generator':'unknown' if r['label']=='spoof' else 'human','split':'test','dataset':'In-the-Wild','language':'English','sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(target/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
(target/'VERIFIED.json').write_text(json.dumps({'verified':True,'audio_files':count,'zip_member_crc_checked':True,'split':'evaluation-only','complete_benchmark_evaluated':False},indent=2));print('In-the-Wild: 31779 recordings verified.',flush=True)
asv=DATA/'datasets/asvspoof2019_la_mirror';wait(asv/'manifest.jsonl');count=0
for line in (asv/'manifest.jsonl').read_text().splitlines():
    r=json.loads(line);assert hashlib.sha256((asv/r['path']).read_bytes()).hexdigest()==r['sha256'];count+=1
assert count==121461
(asv/'ALL-FILES-VERIFIED.json').write_text(json.dumps({'audio_files':count,'all_manifest_sha256_checked':True,'complete_benchmark_evaluated':False},indent=2));print('ASVspoof mirror: 121461 recordings verified.',flush=True)

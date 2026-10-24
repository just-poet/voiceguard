"""Create a portable source/demo package, without redistributing benchmark corpora."""
import hashlib,json,os,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('VOICEGUARD_DATA','D:/VoiceGuardData'))

def main():
    demo=ROOT/'data/partials';demo.mkdir(parents=True,exist_ok=True)
    manifest=DATA/'partials/manifest.jsonl';selected=[];seen=set()
    for line in manifest.read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        if r['control_type'] in seen:continue
        seen.add(r['control_type']);source=manifest.parent/r['path'];shutil.copy2(source,demo/source.name)
        selected.append({**r,'path':source.name})
    (demo/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected),encoding='utf-8')
    (demo/'ATTRIBUTION.txt').write_text('Constructed from SherryT997/IndicTTS-Deepfake-Challenge-Data, revision 57347517658ae989597d8cef303cffb647ed2434, CC BY 4.0. https://huggingface.co/datasets/SherryT997/IndicTTS-Deepfake-Challenge-Data\nChanges: RMS matching, repeat/crop, crossfaded splices. Full lineage and changes are in manifest.jsonl. These are evaluation proxies, not natural calls.\n')
    files=[]
    for directory in ['voiceguard','callguard','frontend/src','frontend/public','frontend/dist','models/aasist','reports/voiceguard','docs/voiceguard','data/demo','data/partials','tools']:
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    names=['README.md','START_VOICEGUARD.cmd','START_CALLGUARD.cmd','start.ps1','setup-voiceguard.ps1','Dockerfile','compose.yaml','.dockerignore','requirements-lock.txt','requirements-voiceguard.txt','requirements-training.txt','requirements-dev.txt','frontend/package.json','frontend/package-lock.json','frontend/tsconfig.json','frontend/vite.config.ts','frontend/index.html','data/indictts/PROVENANCE.json','data/indictts/SOURCE_README.md','models/frozen/aasist-phase1-calibration.json']
    files.extend(ROOT/name for name in names if (ROOT/name).exists())
    files.append(ROOT/'requirements.txt')
    files.extend(ROOT/'tests'/name for name in ['test_voiceguard.py','voiceguard-worklet.test.cjs','voiceguard-stream-interface.test.mjs'])
    files.append(ROOT/'models/frozen/aasist-phase1.pth')
    for name in ['requirements-voiceguard-transitive.txt','reports/voiceguard/submission.html']:
        if (ROOT/name).exists():files.append(ROOT/name)
    files=sorted(set(files));output=ROOT/'output/VoiceGuard_Submission.zip';output.parent.mkdir(exist_ok=True)
    receipt={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,Path('VoiceGuard')/p.relative_to(ROOT))
        z.writestr('VoiceGuard/PACKAGE_SHA256.json',json.dumps(receipt,indent=2))
    with zipfile.ZipFile(output) as z:assert z.testzip() is None
    print(json.dumps({'path':str(output),'bytes':output.stat().st_size,'files':len(files),'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))
if __name__=='__main__':main()

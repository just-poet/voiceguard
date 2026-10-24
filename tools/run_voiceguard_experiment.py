"""Run the authorized training/evaluation after the bounded subset is ready."""
import json,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
manifest=Path('D:/VoiceGuardData/subsets/asvspoof2019_la/manifest.jsonl')
started=time.time()
while not manifest.exists():
    if time.time()-started>7200:raise TimeoutError('ASVspoof subset not ready after two hours; inspect download logs')
    time.sleep(5)
manifests=['data/indictts/manifest.jsonl',str(manifest)]
subprocess.run([sys.executable,'-u','-m','voiceguard.train',*manifests],cwd=ROOT,check=True)
shutil.copy2(ROOT/'models/aasist/weights.pth',ROOT/'models/frozen/aasist-phase1.pth')
calibration=json.loads((ROOT/'models/aasist/calibration.json').read_text())
(ROOT/'models/frozen/aasist-phase1-calibration.json').write_text(json.dumps(calibration['baseline'],indent=2))
subprocess.run([sys.executable,'-u','-m','voiceguard.evaluate',*manifests,'D:/VoiceGuardData/subsets/in_the_wild/manifest.jsonl','--partials','D:/VoiceGuardData/partials/manifest.jsonl'],cwd=ROOT,check=True)
evaluation=json.loads((ROOT/'reports/voiceguard/evaluation.json').read_text())
(ROOT/'reports/voiceguard/phase1-reference.json').write_text(json.dumps({'name':'Newly frozen official AASIST-L reference, not historical contest results','checkpoint':evaluation['configuration']['model']['baseline_sha256'],'datasets':{k:v['baseline'] for k,v in evaluation['groups'].items()}},indent=2))
print('Training and evaluation complete.',flush=True)

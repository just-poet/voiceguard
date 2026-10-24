"""Restore pinned official assets with exact SHA256 verification."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];receipt=json.loads((ROOT/'models/aasist/PROVENANCE.json').read_text());base='https://raw.githubusercontent.com/clovaai/aasist/'+receipt['revision']+'/'
files={'weights.pth':('models/weights/AASIST-L.pth','models/aasist/weights.pth'),'architecture.py':('models/AASIST.py','voiceguard/vendor/aasist.py'),'config.json':('config/AASIST-L.conf','models/aasist/config.json'),'LICENSE':('LICENSE','voiceguard/vendor/AASIST-LICENSE')}
for key,(remote,local) in files.items():
    target=ROOT/local
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==receipt['files'][key]:print(local,'verified');continue
    data=urllib.request.urlopen(base+remote,timeout=60).read()
    if hashlib.sha256(data).hexdigest()!=receipt['files'][key]:raise ValueError('Source checksum mismatch: '+remote)
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);print(local,'restored')

import json,sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from voiceguard.model import Engine
from voiceguard.audio import ffmpeg_path
engine=Engine();print(json.dumps({'checkpoint_verified':True,'engines':engine.status()['engines'],'device':str(engine.device),'ffmpeg':ffmpeg_path(),'frontend_built':Path('frontend/dist/index.html').is_file()},indent=2))
subprocess.run([ffmpeg_path(),'-version'],check=True,stdout=subprocess.DEVNULL)

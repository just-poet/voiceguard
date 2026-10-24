"""Verify an extracted submission with no external dataset access."""
import hashlib,json,os,subprocess,sys,tempfile,time,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DATA=Path(os.environ.get('VOICEGUARD_DATA','D:/VoiceGuardData'))
target=Path(tempfile.mkdtemp(prefix='voiceguard-package-',dir=DATA/'temp'))
with zipfile.ZipFile(ROOT/'output/VoiceGuard_Submission.zip') as archive:
    for member in archive.namelist():
        if not (target/member).resolve().is_relative_to(target.resolve()):raise ValueError('Unsafe package member')
    archive.extractall(target)
project=target/'VoiceGuard';receipt=json.loads((project/'PACKAGE_SHA256.json').read_text())
for name,digest in receipt.items():assert hashlib.sha256((project/name).read_bytes()).hexdigest()==digest
env={**os.environ,'VOICEGUARD_DATA':str(target/'empty-data'),'VOICEGUARD_THREADS':'6'}
log=(target/'server.log').open('w');process=subprocess.Popen([sys.executable,'-m','uvicorn','voiceguard.server:app','--host','127.0.0.1','--port','8768','--ws-max-size','32012','--ws-max-queue','2'],cwd=project,env=env,stdout=log,stderr=log)
def get(path):return urllib.request.urlopen('http://127.0.0.1:8768'+path,timeout=30)
try:
    for _ in range(90):
        try:status=json.load(get('/api/status'));break
        except OSError:
            if process.poll() is not None:raise RuntimeError('Packaged server exited; see '+str(target/'server.log'))
            time.sleep(.5)
    else:raise TimeoutError('Package server startup timed out')
    assert status['model']['backend']=='onnx_cpu' and 'candidate' in status['model']['engines'];assert b'VoiceGuard' in get('/').read()
    examples=json.load(get('/api/examples'));assert len(examples)==10;results=[]
    for example in examples:
        audio=get('/api/examples/'+example['id']).read();request=urllib.request.Request('http://127.0.0.1:8768/api/analyze?model=candidate',data=audio,headers={'Content-Type':'application/octet-stream'})
        result=json.load(urllib.request.urlopen(request,timeout=60));assert result['summary']['score'] is not None and result['raw_audio_retained'] is False
        results.append({'example':example['name'],'score':result['summary']['score'],'verdict':result['summary']['verdict'],'windows':len(result['bins'])})
    report={'passed':True,'package_files_verified':len(receipt),'package_sha256':hashlib.sha256((ROOT/'output/VoiceGuard_Submission.zip').read_bytes()).hexdigest(),'extracted_directory':str(project),'python':sys.executable,'external_datasets_available':False,'fresh_environment':False,'environment_note':'Uses the tested D: environment; package source and demo assets are independently extracted.','examples':results,'backend':status['model']['backend']}
    (ROOT/'reports/voiceguard/package-verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
finally:
    process.terminate();process.wait(timeout=20);log.close()

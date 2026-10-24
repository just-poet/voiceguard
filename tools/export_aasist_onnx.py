"""Optional CPU export of the unchanged official encoder, with numerical checks."""
import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import onnx,onnxruntime as ort,torch
from voiceguard.model import MODEL_ROOT,N_SAMPLES,fit_length,sha
from voiceguard.vendor.aasist import Model
from voiceguard.audio import decode,SR
torch.set_num_threads(6)
model=Model(json.loads((MODEL_ROOT/'config.json').read_text())['model_config']);model.load_state_dict(torch.load(MODEL_ROOT/'weights.pth',map_location='cpu',weights_only=True));model.eval()
path=MODEL_ROOT/'encoder.onnx'
with torch.inference_mode():
    torch.onnx.export(model,(torch.zeros((1,N_SAMPLES)),),str(path),input_names=['audio'],output_names=['embedding','logits'],opset_version=17,dynamo=False,external_data=False)
onnx.checker.check_model(onnx.load(path));print('Exported and checked ONNX graph',flush=True)
def session(threads):
    options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1;options.add_session_config_entry('session.intra_op.allow_spinning','0');options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
runtime=session(4);inputs=[np.zeros(N_SAMPLES,dtype=np.float32),np.random.default_rng(42).normal(0,.04,N_SAMPLES).astype(np.float32)];source_ids=[];groups={}
for manifest in [Path('data/indictts/manifest.jsonl'),Path('D:/VoiceGuardData/subsets/asvspoof2019_la/manifest.jsonl')]:
    for line in manifest.read_text().splitlines():
        row=json.loads(line);key=(row['dataset'],row.get('language'),row['label'])
        if row['split']!='dev' or groups.get(key,0)>=2:continue
        audio=decode(manifest.parent/row['path'],max_seconds=60)
        if len(audio)<2*SR:continue
        groups[key]=groups.get(key,0)+1;source_ids.append(row['source_id']);inputs.extend([fit_length(audio[:2*SR]),fit_length(audio[:4*SR])])
maximum=0.
for i,x in enumerate(inputs):
    x=x[None].astype(np.float32)
    with torch.inference_mode():reference=model(torch.from_numpy(x))
    actual=runtime.run(None,{'audio':x})
    for a,b in zip(actual,reference):
        b=b.numpy();maximum=max(maximum,float(np.max(abs(a-b))));np.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-4)
    if i%10==0:print('Parity',i,'/',len(inputs),flush=True)
timings=[];x=inputs[1][None]
for threads in (1,2,4,6):
    runtime=session(threads)
    for _ in range(3):runtime.run(None,{'audio':x})
    values=[]
    for _ in range(10):
        start=time.perf_counter();runtime.run(None,{'audio':x});values.append((time.perf_counter()-start)*1000)
    timings.append({'threads':threads,'p50_ms':float(np.median(values)),'p95_ms':float(np.percentile(values,95))});print(timings[-1],flush=True)
receipt={'sha256':sha(path),'source_weights_sha256':sha(MODEL_ROOT/'weights.pth'),'source_config_sha256':sha(MODEL_ROOT/'config.json'),'onnxruntime_version':ort.__version__,'onnx_version':onnx.__version__,'opset':17,'fixed_input_shape':[1,N_SAMPLES],'quantized':False,'parity_cases':len(inputs),'development_source_ids':source_ids,'max_absolute_tensor_error':maximum,'tolerance':{'atol':1e-4,'rtol':1e-4},'timings':timings,'recommended_threads':min(timings,key=lambda r:r['p95_ms'])['threads']}
(MODEL_ROOT/'ONNX.json').write_text(json.dumps(receipt,indent=2));Path('reports/voiceguard/onnx-parity.json').write_text(json.dumps(receipt,indent=2));print('ONNX parity and CPU timing complete.',flush=True)

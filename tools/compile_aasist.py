"""Build a local fixed-input TorchScript cache and verify numeric parity."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from voiceguard.vendor.aasist import Model
from voiceguard.model import MODEL_ROOT,N_SAMPLES,sha
torch.set_num_threads(4);torch.manual_seed(42)
network=Model(json.loads((MODEL_ROOT/'config.json').read_text())['model_config']);network.load_state_dict(torch.load(MODEL_ROOT/'weights.pth',weights_only=True,map_location='cpu'));network.eval()
with torch.inference_mode():
    sample=torch.randn(1,N_SAMPLES)*.05;compiled=torch.jit.freeze(torch.jit.trace(network,sample,check_trace=False))
    errors=[]
    for sample in (sample,torch.randn(1,N_SAMPLES)*.1):
        errors.append(float(torch.max(torch.abs(network(sample)[1]-compiled(sample)[1]))))
    if max(errors)>1e-4:raise ValueError('Compiled output differs from official eager inference')
    compiled.save(str(MODEL_ROOT/'compiled.pt'))
(MODEL_ROOT/'COMPILED.json').write_text(json.dumps({'sha256':sha(MODEL_ROOT/'compiled.pt'),'source_weights_sha256':sha(MODEL_ROOT/'weights.pth'),'input_samples':N_SAMPLES,'torch':torch.__version__,'method':'Fixed-input torch.jit.trace + freeze','max_logit_difference':max(errors)},indent=2))
print('Compiled cache verified; maximum difference',max(errors))

"""Official AASIST-L and a trained, frozen-encoder adaptation head."""
import hashlib,json,os,threading,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits
from .vendor.aasist import Model
from .audio import SR,HOP,quality
from callguard.features import extract

ROOT=Path(__file__).resolve().parents[1];MODEL_ROOT=ROOT/'models/aasist';N_SAMPLES=64600
torch.set_num_threads(int(os.environ.get('VOICEGUARD_THREADS','4')))
# LFCC matrix products are small; a 16-thread BLAS pool competes with AASIST.
_blas_limit=threadpool_limits(limits=1,user_api='blas')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fit_length(x):
    x=np.asarray(x,dtype=np.float32)
    if not len(x):raise ValueError('Empty signal')
    return x[:N_SAMPLES] if len(x)>=N_SAMPLES else np.tile(x,(N_SAMPLES+len(x)-1)//len(x))[:N_SAMPLES]

class AdaptationHead(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(248,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,1))
    def forward(self,x):return self.net(x).squeeze(-1)

class Engine:
    def __init__(self,device='cpu',load_candidate=True,backend='auto'):
        self.device=torch.device(device);self.lock=threading.Lock()
        provenance=json.loads((MODEL_ROOT/'PROVENANCE.json').read_text())
        if sha(MODEL_ROOT/'weights.pth')!=provenance['files']['weights.pth']:raise ValueError('Checkpoint checksum mismatch')
        if sha(ROOT/'voiceguard/vendor/aasist.py')!=provenance['files']['architecture.py']:raise ValueError('Architecture checksum mismatch')
        if sha(MODEL_ROOT/'config.json')!=provenance['files']['config.json']:raise ValueError('Configuration checksum mismatch')
        config=json.loads((MODEL_ROOT/'config.json').read_text())['model_config']
        self.network=Model(config).to(self.device)
        self.network.load_state_dict(torch.load(MODEL_ROOT/'weights.pth',map_location=self.device,weights_only=True));self.network.eval()
        compiled=MODEL_ROOT/'compiled.pt';receipt=MODEL_ROOT/'COMPILED.json'
        if str(self.device)=='cpu' and compiled.exists() and receipt.exists():
            expected=json.loads(receipt.read_text())
            if sha(compiled)!=expected['sha256'] or sha(MODEL_ROOT/'weights.pth')!=expected['source_weights_sha256']:raise ValueError('Compiled checkpoint checksum mismatch')
            self.network=torch.jit.load(str(compiled),map_location=self.device);self.network.eval()
        self.ort=None;self.backend='pytorch';self.ort_threads=None
        onnx_file=MODEL_ROOT/'encoder.onnx';onnx_receipt=MODEL_ROOT/'ONNX.json'
        if str(self.device)=='cpu' and backend!='torch' and onnx_file.exists() and onnx_receipt.exists():
            try:
                import onnxruntime as ort
            except ImportError:
                if backend=='onnx':raise
            else:
                expected=json.loads(onnx_receipt.read_text())
                if sha(onnx_file)!=expected['sha256'] or sha(MODEL_ROOT/'weights.pth')!=expected['source_weights_sha256'] or sha(MODEL_ROOT/'config.json')!=expected['source_config_sha256']:raise ValueError('ONNX checkpoint checksum mismatch')
                options=ort.SessionOptions();self.ort_threads=int(os.environ.get('VOICEGUARD_ORT_THREADS',str(expected['recommended_threads'])));options.intra_op_num_threads=self.ort_threads;options.inter_op_num_threads=1;options.add_session_config_entry('session.intra_op.allow_spinning','0');options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.ort=ort.InferenceSession(str(onnx_file),sess_options=options,providers=['CPUExecutionProvider']);self.backend='onnx_cpu'
        self.calibration={};self.head=None;self.training={}
        self.local_head=None;self.local_threshold=0.50
        det_file=ROOT/'models/detector.json'
        if det_file.exists():
            try:
                det=json.loads(det_file.read_text(encoding='utf-8'))
                self.local_head=det.get('local_head')
                self.local_threshold=float(det.get('local_threshold',0.50))
            except Exception:pass
        if load_candidate:self.reload_candidate()
        # Warm the fixed-size execution graph before advertising readiness.
        with torch.inference_mode():
            for _ in range(3):
                if self.ort:self.ort.run(None,{'audio':np.zeros((1,N_SAMPLES),dtype=np.float32)})
                else:self.network(torch.zeros((1,N_SAMPLES),device=self.device))
    def reload_candidate(self):
        calibration=MODEL_ROOT/'calibration.json'
        if calibration.exists():self.calibration=json.loads(calibration.read_text())
        candidate=MODEL_ROOT/'candidate.pt'
        if candidate.exists():
            state=torch.load(candidate,map_location='cpu',weights_only=True)
            self.head=AdaptationHead().to(self.device);self.head.load_state_dict(state['state_dict']);self.head.eval()
            self.mean=np.asarray(state['mean']);self.scale=np.asarray(state['scale']);self.training=state['provenance']
    def status(self):
        return {'ready':True,'name':'Official AASIST-L + trained adaptation','engines':['baseline']+(['candidate'] if self.head else []),'device':str(self.device),'backend':self.backend,'onnx_threads':self.ort_threads,'threads':torch.get_num_threads(),'sample_rate':SR,'input_samples':N_SAMPLES,'context_seconds':N_SAMPLES/SR,'warmup_seconds':2,'baseline_sha256':sha(MODEL_ROOT/'weights.pth'),'candidate_sha256':sha(MODEL_ROOT/'candidate.pt') if self.head else None,'calibration':self.calibration,'training':self.training,'score_kind':'Uncalibrated logit score; higher means more synthetic evidence','localization':'Dual-timescale: AASIST-L global clip analysis + 0.5-second local segmenter for partial fakes'}
    def features(self,x):
        original=np.asarray(x)
        with self.lock,torch.inference_mode():
            if self.ort:embedding,logits=self.ort.run(None,{'audio':fit_length(original).copy()[None]})
            else:
                embedding,logits=self.network(torch.from_numpy(fit_length(original).copy()).unsqueeze(0).to(self.device));embedding=embedding.cpu().numpy();logits=logits.cpu().numpy()
        features=np.concatenate([embedding[0],extract(original)])
        return features,float(logits[0,0]-logits[0,1]),logits[0].tolist()
    def score_local(self,chunk):
        if self.local_head is None or len(chunk)<400:return None
        feat=extract(chunk)
        scaled=((feat-np.asarray(self.local_head['mean']))/np.asarray(self.local_head['scale'])).astype(np.float32)
        probs=[]
        for tree in self.local_head['trees']:
            node=0
            while tree['left'][node]!=-1:
                node=tree['left'][node] if scaled[tree['feature'][node]]<=tree['threshold'][node] else tree['right'][node]
            probs.append(tree['probability'][node])
        return float(np.mean(probs))
    def score(self,x,engine='candidate',warmup=True):
        start=time.perf_counter();q=quality(x)
        if len(x)<2*SR and warmup:
            return {'score':None,'verdict':'Inconclusive','state':'warmup','quality':q,'processing_ms':round((time.perf_counter()-start)*1000,2),'context_seconds':len(x)/SR}
        if engine not in ('baseline','candidate'):raise ValueError('Unknown model')
        if engine=='candidate' and self.head is None:raise ValueError('Candidate has not been trained')
        features,baseline,logits=self.features(x)
        score=baseline
        if engine=='candidate':
            with torch.inference_mode():head_s=float(self.head(torch.tensor((features-self.mean)/self.scale,dtype=torch.float32,device=self.device).unsqueeze(0))[0].cpu())
            score=head_s
        calibration=self.calibration.get(engine);threshold=calibration.get('threshold') if calibration else None
        verdict='Inconclusive';reasons=list(q['reasons'])
        if calibration is None:reasons.append('No development-fixed threshold available')
        elif abs(score-threshold)<calibration.get('margin',0):reasons.append('Score close to the development threshold')
        elif q['usable']:verdict='Likely synthetic' if score>=threshold else 'Likely bona fide'
        return {'score':score,'verdict':verdict,'state':'scored','quality':q,'reasons':reasons,'threshold':threshold,'processing_ms':round((time.perf_counter()-start)*1000,2),'context_seconds':min(len(x),N_SAMPLES)/SR,'repeated_padding_seconds':max(0,N_SAMPLES-len(x))/SR,'baseline_logits':logits if engine=='baseline' else None}
    def analyze(self,x,engine='candidate'):
        started=time.perf_counter();summary=self.score(x,engine);bins=[]
        for offset in range(0,len(x),HOP):
            end=min(offset+HOP,len(x));chunk=x[offset:end];window=x[max(0,end-N_SAMPLES):end]
            b=self.score(window,engine)
            local_s=self.score_local(chunk)
            if local_s is not None and b['state']=='scored':
                b['local_score']=round(local_s,4)
                if local_s>=self.local_threshold and b['quality']['usable']:
                    b['verdict']='Likely synthetic'
                    b['reasons']=['Synthetic speech artifacts detected in 0.5s segment']
            bins.append({'start':offset/SR,'end':end/SR,**b})
        synth_flags=[(b.get('local_score') is not None and b['local_score']>=self.local_threshold) or b.get('verdict')=='Likely synthetic' for b in bins]
        clustered_synthetic=any(sum(synth_flags[i:i+3])>=2 for i in range(len(synth_flags)-2)) if len(synth_flags)>=3 else (sum(synth_flags)>=2)
        if (clustered_synthetic or sum(synth_flags)>=4) and summary['verdict'] in ('Likely bona fide','Inconclusive'):
            summary['verdict']='Likely synthetic'
            summary['reasons']=[f'Partly faked speech: {sum(synth_flags)} synthetic 0.5s segment(s) detected']
        return {'schema_version':2,'engine':engine,'duration':len(x)/SR,'summary':summary,'bins':bins,'quality':quality(x),'waveform':[float(np.max(np.abs(v))) for v in np.array_split(x,min(240,len(x)))],'processing_ms':(time.perf_counter()-started)*1000,'model':self.status(),'note':'AASIST uses first 64600 samples, repeating shorter inputs. High-resolution 0.5s local segmenter pinpoints partly faked speech. Acoustic authenticity does not establish caller identity or scam intent.'}


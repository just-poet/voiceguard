"""Train a real PyTorch head over frozen AASIST-L embeddings plus LFCCs."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from .model import Engine,AdaptationHead,MODEL_ROOT,sha
from .audio import decode,transform,SR
from .datasets import load_manifest,validate_lineage
from callguard.metrics import eer

def main():
    p=argparse.ArgumentParser();p.add_argument('manifests',nargs='+');p.add_argument('--cache',default='D:/VoiceGuardData/cache/features');p.add_argument('--device',default='cpu');p.add_argument('--epochs',type=int,default=80);p.add_argument('--features-only',action='store_true');a=p.parse_args()
    np.random.seed(42);torch.manual_seed(42);rows=sum([load_manifest(m) for m in a.manifests],[]);validate_lineage(rows)
    # Existing adaptation files must not consume RNG or affect a fresh training run.
    engine=Engine(a.device,load_candidate=False,backend='torch');cache=Path(a.cache);cache.mkdir(parents=True,exist_ok=True)
    train=[];dev=[];excluded=[];started=time.time();seen=set()
    selected=[r for r in rows if r['split'] in ('train','dev')]
    for index,r in enumerate(selected):
        if r['sha256'] in seen:continue
        seen.add(r['sha256'])
        try:x=decode(r['path'],max_seconds=60)
        except ValueError as error:excluded.append({'source':r['source_id'],'reason':str(error)});continue
        if len(x)<2*SR:excluded.append({'source':r['source_id'],'reason':'less than 2 seconds'});continue
        x=x[:6*SR]
        # Class-independent augmentation: source hash, not label, chooses the channel.
        conditions=['clean']
        seed=int(r['sha256'][:8],16)
        if r['split']=='train':conditions.append(['mulaw','alaw','noise10','telephone_noise10'][seed%4])
        for condition in conditions:
            key=hashlib.sha256(f"aasist64600-lfcc-v2:{r['sha256']}:{condition}:{seed}".encode()).hexdigest();file=cache/(key+'.npz')
            if file.exists():
                with np.load(file,allow_pickle=False) as saved:features=saved['features'];baseline=float(saved['baseline'])
            else:
                features,baseline,_=engine.features(transform(x,condition,seed));np.savez(file,features=features,baseline=baseline)
            item={'features':features,'baseline':baseline,'label':r['label'],'row':r,'condition':condition}
            (train if r['split']=='train' else dev).append(item)
        if index%50==0:print(f'Features {index}/{len(selected)}, elapsed {time.time()-started:.1f}s',flush=True)
    if a.features_only:print('Feature cache ready; no model or thresholds changed',flush=True);return
    if {r['label'] for r in train}!={0,1} or {r['label'] for r in dev}!={0,1}:raise ValueError('Training and development each require both labels')
    X=np.stack([r['features'] for r in train]);mean=X.mean(0);scale=X.std(0);scale[scale<1e-6]=1
    tx=torch.tensor((X-mean)/scale,dtype=torch.float32,device=a.device);ty=torch.tensor([r['label'] for r in train],dtype=torch.float32,device=a.device)
    dx=torch.tensor((np.stack([r['features'] for r in dev])-mean)/scale,dtype=torch.float32,device=a.device);dy=np.array([r['label'] for r in dev])
    head=AdaptationHead().to(a.device);optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    weight=(len(ty)-ty.sum())/ty.sum();loss_fn=nn.BCEWithLogitsLoss(pos_weight=weight)
    best=float('inf');best_state=None;history=[];patience=0
    for epoch in range(a.epochs):
        head.train();order=torch.randperm(len(tx),device=a.device);losses=[]
        for indices in order.split(64):
            optimizer.zero_grad();loss=loss_fn(head(tx[indices]),ty[indices]);loss.backward();optimizer.step();losses.append(float(loss.detach().cpu()))
        head.eval()
        with torch.inference_mode():scores=head(dx).cpu().numpy()
        dev_eer=eer(dy,scores)['eer_percent'];history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'development_eer_percent':dev_eer,'development_auc':float(roc_auc_score(dy,scores))})
        if dev_eer<best:
            best=dev_eer;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};patience=0
        else:patience+=1
        print(json.dumps(history[-1]),flush=True)
        if patience>=15:break
    head.load_state_dict(best_state);head.eval()
    with torch.inference_mode():candidate_scores=head(dx).cpu().numpy()
    baseline_scores=np.array([r['baseline'] for r in dev]);calibration={}
    for name,scores in [('baseline',baseline_scores),('candidate',candidate_scores)]:
        metric=eer(dy,scores);threshold=metric['threshold'];margin=float(np.quantile(np.abs(scores-threshold),.1))
        calibration[name]={'threshold':threshold,'margin':margin,'development_eer_percent':metric['eer_percent'],'development_auc':float(roc_auc_score(dy,scores)),'clips':len(dev),'probability_calibrated':False,'method':'EER operating threshold; nearest 10% of development scores form declared abstention margin'}
    provenance={'method':'Frozen official AASIST-L 160-dimensional embedding + 88 LFCC statistics; trained PyTorch 248-64-1 MLP. Encoder weights unchanged.','pretrained_on':'ASVspoof2019 LA training set according to official repository; no world-first or universal unseen-generator claim','seed':42,'epochs':len(history),'train_windows':len(train),'development_clips':len(dev),'source_manifests':a.manifests,'train_hashes':sorted({r['row']['sha256'] for r in train}),'development_hashes':sorted({r['row']['sha256'] for r in dev}),'baseline_sha256':sha(MODEL_ROOT/'weights.pth'),'augmentation':'One clean and one hash-selected phone/noise condition per training source; never applied to development labels or test training','elapsed_seconds':time.time()-started,'device':a.device}
    torch.save({'state_dict':best_state,'mean':mean.tolist(),'scale':scale.tolist(),'provenance':provenance},MODEL_ROOT/'candidate.pt')
    (MODEL_ROOT/'calibration.json').write_text(json.dumps(calibration,indent=2));(MODEL_ROOT/'TRAINING.json').write_text(json.dumps({'provenance':provenance,'epochs':history,'excluded':excluded,'calibration':calibration},indent=2))
    out=Path('reports/voiceguard');out.mkdir(parents=True,exist_ok=True)
    (out/'training.json').write_text((MODEL_ROOT/'TRAINING.json').read_text())
    (out/'development-scores.json').write_text(json.dumps([{'source_id':r['row']['source_id'],'dataset':r['row']['dataset'],'label':int(dy[i]),'baseline_score':float(baseline_scores[i]),'candidate_score':float(candidate_scores[i])} for i,r in enumerate(dev)],indent=2))
    print(json.dumps(calibration,indent=2),flush=True)
if __name__=='__main__':main()

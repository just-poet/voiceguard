"""Construct labelled partials and matched sham/full controls from test sources only."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import soundfile as sf
from .audio import decode,SR
from .datasets import load_manifest

def normalize(x):return np.clip(x*.05/max(float(np.sqrt(np.mean(x*x))),1e-6),-.95,.95)
def span(x,length):return np.tile(x,(length+len(x)-1)//len(x))[:length]
def splice(base,donor,start,end):
    result=base.copy();insert=normalize(span(donor,end-start));fade=min(160,(end-start)//4);blend=np.ones(end-start);blend[:fade]=np.linspace(0,1,fade);blend[-fade:]=np.linspace(1,0,fade)
    result[start:end]=result[start:end]*(1-blend)+insert*blend;return result

def main():
    p=argparse.ArgumentParser();p.add_argument('manifests',nargs='+');p.add_argument('--output',default='D:/VoiceGuardData/partials');p.add_argument('--count',type=int,default=100);a=p.parse_args()
    rows=sum([load_manifest(m) for m in a.manifests],[]);rows=[r for r in rows if r['split']=='test'];pool=[]
    for r in rows:
        try:x=decode(r['path'],max_seconds=60)
        except ValueError:continue
        if len(x)>=2*SR:pool.append((r,normalize(x)))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(42);manifest=[]
    languages=sorted({r.get('language','English') for r,x in pool})
    for i in range(a.count):
        language=languages[i%len(languages)];genuine=[(r,x) for r,x in pool if r['label']==0 and r.get('language','English')==language];fake=[(r,x) for r,x in pool if r['label']==1 and r.get('language','English')==language]
        if not genuine or not fake:raise ValueError('Each language needs both classes')
        base_r,base_x=genuine[int(rng.integers(len(genuine)))];matched=[v for v in fake if v[0]['speaker']==base_r['speaker'] and base_r['speaker']!='unknown'];fake_r,fake_x=(matched or fake)[int(rng.integers(len(matched or fake)))];sham_choices=[v for v in genuine if v[0]['sha256']!=base_r['sha256']];sham_r,sham_x=sham_choices[int(rng.integers(len(sham_choices)))]
        start=float(rng.choice([2.,2.5,3.,3.5]));duration=float(rng.choice([.5,1.,1.5,2.]));end=min(5.5,start+duration)
        cases=[('partial',base_r,base_x,fake_r,fake_x,1,[[start,end]]),('sham_real_to_real',base_r,base_x,sham_r,sham_x,0,[])]
        if i<20:cases.extend([('full_real',base_r,base_x,base_r,base_x,0,[]),('full_synthetic',fake_r,fake_x,fake_r,fake_x,1,[[0,6.]])])
        for kind,left,lx,right,rx,label,intervals in cases:
            audio=splice(normalize(span(lx,6*SR)),rx,int(start*SR),int(end*SR));name=f'{kind}_{i:04d}.wav';sf.write(out/name,audio,SR,subtype='PCM_16')
            manifest.append({'path':name,'label':label,'speaker':'unknown','generator':'constructed-proxy','language':language,'dataset':'Partial proxy','split':'test','source_id':name,'control_type':kind,'fake_segments':intervals,'source_ids':[left['source_id'],right['source_id']],'source_hashes':[left['sha256'],right['sha256']],'source_speakers':[left['speaker'],right['speaker']],'same_speaker_verified':left['speaker']==right['speaker']!='unknown','seed':42,'edit_interval':[start,end],'transformations':{'sample_rate':SR,'rms':.05,'crossfade_ms':10,'base_repeat_padding':len(lx)<6*SR,'processing_matched_across_classes':True},'sha256':hashlib.sha256((out/name).read_bytes()).hexdigest()})
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest),encoding='utf-8');print(f'{len(manifest)} examples; {a.count} partials and {a.count} shams')
if __name__=='__main__':main()

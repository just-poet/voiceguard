"""Bounded content-based decoding; temporary recordings are always removed."""
from pathlib import Path
import shutil,subprocess,tempfile
import numpy as np
import soundfile as sf
from callguard.audio import SR,HOP,resample,telephone,add_noise

MAX_BYTES=8*1024*1024
def ffmpeg_path():
    found=shutil.which('ffmpeg')
    if found:return found
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()

def decode(source,max_seconds=10):
    data=source if isinstance(source,bytes) else Path(source).read_bytes()
    if not data or len(data)>MAX_BYTES:raise ValueError('Audio must be nonempty and at most 8 MB.')
    # No supplied filename or external URL enters the command.
    with tempfile.TemporaryDirectory(prefix='voiceguard-') as directory:
        path=Path(directory)/'recording';path.write_bytes(data)
        args=[ffmpeg_path(),'-hide_banner','-nostdin','-v','error','-protocol_whitelist','file,pipe','-format_whitelist','wav,flac,mp3,ogg,matroska,webm,mov,aac','-i',str(path),'-map','0:a:0','-vn','-sn','-dn','-t',str(max_seconds+1/SR),'-ac','1','-ar',str(SR),'-f','f32le','pipe:1']
        try:r=subprocess.run(args,capture_output=True,timeout=20,check=False)
        except subprocess.TimeoutExpired as e:raise ValueError('Audio decoding exceeded 20 seconds.') from e
        if r.returncode or not r.stdout:raise ValueError('Cannot decode audio content. Use WAV, FLAC, MP3, OGG, WebM or M4A.')
        if len(r.stdout)>int(max_seconds*SR)*4:raise ValueError(f'Audio must be at most {max_seconds} seconds.')
        x=np.frombuffer(r.stdout,dtype='<f4').astype(np.float64)
        if not np.isfinite(x).all():raise ValueError('Invalid audio samples.')
        peak=float(np.max(np.abs(x))) if len(x) else 0.0
        if peak>0.8:x=x*(0.6/peak)
        return x

def quality(x):
    x=np.asarray(x);rms=float(np.sqrt(np.mean(x*x))) if len(x) else 0
    frames=[x[i:i+320] for i in range(0,len(x),320)]
    active=sum(len(f) for f in frames if np.sqrt(np.mean(f*f))>.008)/SR
    clipped=float(np.mean(np.abs(x)>=.999)) if len(x) else 0
    reasons=[]
    if rms<.004:reasons.append('Very low audio level / silence')
    if active<1:reasons.append('Less than one second of energy-active audio')
    if clipped>.05:reasons.append('Severe clipping')
    return {'rms_dbfs':round(20*np.log10(max(rms,1e-8)),1),'active_seconds':round(active,3),'clipped_fraction':round(clipped,4),'usable':not reasons,'reasons':reasons,'method':'20 ms energy gate; not a learned speech detector'}

CONDITIONS=('clean','resample8k','mulaw','alaw','noise20','noise10','telephone_noise10')
def transform(x,name,seed=42):
    if name=='clean':return x.copy()
    if name=='resample8k':return resample(resample(x,SR,8000),8000,SR)[:len(x)]
    if name in ('mulaw','alaw'):return telephone(x,name)
    if name.startswith('noise'):return add_noise(x,float(name[5:]),seed)
    if name=='telephone_noise10':return add_noise(telephone(x,'mulaw'),10,seed)
    raise ValueError('Unknown audio condition')

TRANSFORM_CONFIG={'resample8k':'polyphase 16k->8k->16k, no bandpass','mulaw':'300-3400 Hz bandpass; 8 kHz G.711 mu-law codewords and decode','alaw':'300-3400 Hz bandpass; 8 kHz G.711 A-law codewords and decode','noise20':'Gaussian noise at 20 dB signal-relative SNR','noise10':'Gaussian noise at 10 dB signal-relative SNR','telephone_noise10':'8 kHz + 300-3400 Hz + G.711 mu-law + 10 dB noise','seed':42}

def reference_codec(x,condition,seed=42):
    """Encode a real 8-bit G.711 WAV container with libsndfile, then decode it."""
    from io import BytesIO
    from scipy.signal import butter,sosfilt
    if condition not in ('g711_mulaw_wav','g711_alaw_wav','g711_mulaw_wav_noise10'):raise ValueError('Unknown reference codec')
    low=sosfilt(butter(4,[300,3400],btype='bandpass',fs=8000,output='sos'),resample(x,SR,8000))
    buffer=BytesIO();sf.write(buffer,low,8000,format='WAV',subtype='ALAW' if condition=='g711_alaw_wav' else 'ULAW');buffer.seek(0)
    decoded,rate=sf.read(buffer,dtype='float64');assert rate==8000
    y=resample(decoded,8000,SR)[:len(x)]
    return add_noise(y,10,seed) if condition.endswith('noise10') else y

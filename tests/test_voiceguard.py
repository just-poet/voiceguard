from io import BytesIO
from pathlib import Path
import json,struct,subprocess,tempfile
import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient
from voiceguard.audio import decode,ffmpeg_path,quality,transform,SR,HOP
from voiceguard.model import Engine,fit_length,N_SAMPLES,MODEL_ROOT
from voiceguard.metrics import metrics,bin_label
from voiceguard.datasets import validate_lineage

def wav(seconds=2,rate=16000):
    x=.1*np.sin(2*np.pi*173*np.arange(int(seconds*rate))/rate);buffer=BytesIO();sf.write(buffer,x,rate,format='WAV',subtype='PCM_16');return buffer.getvalue()

def test_decoder_duration_content_and_cleanup():
    before=set(Path(tempfile.gettempdir()).glob('voiceguard-*'))
    assert len(decode(wav(2,48000)))==32000
    with pytest.raises(ValueError):decode(b'not sound')
    with pytest.raises(ValueError):decode(b'ffconcat version 1.0\nfile /etc/private.wav\n')
    with pytest.raises(ValueError):decode(wav(10.2))
    with pytest.raises(ValueError):decode(b'x'*(8*1024*1024+1))
    assert set(Path(tempfile.gettempdir()).glob('voiceguard-*'))==before

@pytest.mark.parametrize('codec,fmt',[('libmp3lame','mp3'),('libopus','webm')])
def test_real_compressed_audio_decoding(codec,fmt):
    encoded=subprocess.run([ffmpeg_path(),'-v','error','-f','wav','-i','pipe:0','-c:a',codec,'-f',fmt,'pipe:1'],input=wav(),capture_output=True,check=True).stdout
    x=decode(encoded);assert abs(len(x)/SR-2)<.15 and quality(x)['usable']

def test_quality_and_padding_are_honest():
    assert not quality(np.zeros(32000))['usable']
    assert not quality(np.ones(32000))['usable']
    x=np.arange(32000,dtype=np.float32);p=fit_length(x)
    assert len(p)==64600;np.testing.assert_array_equal(p[32000:64000],x)
    np.testing.assert_array_equal(fit_length(np.arange(80000)),np.arange(64600))

def test_transform_reproducibility_and_actual_downsample():
    x=decode(wav());y=transform(x,'telephone_noise10',42)
    np.testing.assert_array_equal(y,transform(x,'telephone_noise10',42));assert len(y)==len(x)
    assert not np.allclose(y,x)
    assert not np.allclose(transform(x,'resample8k'),transform(x,'mulaw'))

@pytest.mark.parametrize('subtype,condition',[('ULAW','g711_mulaw_wav'),('ALAW','g711_alaw_wav')])
def test_reference_codec_containers_decode_with_ffmpeg(subtype,condition):
    from voiceguard.audio import reference_codec
    x=np.linspace(-.9,.9,8000);buffer=BytesIO();sf.write(buffer,x,8000,format='WAV',subtype=subtype)
    encoded=buffer.getvalue();buffer.seek(0);assert sf.info(buffer).subtype==subtype
    buffer.seek(0);expected,rate=sf.read(buffer);assert rate==8000
    result=subprocess.run([ffmpeg_path(),'-v','error','-f','wav','-i','pipe:0','-f','f32le','pipe:1'],input=encoded,capture_output=True,check=True)
    np.testing.assert_allclose(np.frombuffer(result.stdout,dtype='<f4'),expected,atol=1e-7)
    speech=decode(wav());actual=reference_codec(speech,condition);assert len(actual)==len(speech) and not np.allclose(actual,speech)

def test_known_metrics_and_zero_false_alarm():
    m=metrics([0,0,1,1],[-2,-1,1,2],0)
    assert m['eer_percent']==0 and m['roc_auc']==1
    assert m['confusion_matrix']==[[2,0],[0,2]] and m['false_alarm_rate']==0 and m['miss_rate']==0
    assert bin_label(2,2.5,[[2.25,3]])
    assert not bin_label(2,2.5,[[2.26,3]])

def test_descendant_and_speaker_leakage():
    a={'sha256':'parent','split':'train','speaker':'a','dataset':'X'}
    b={'sha256':'edit','source_hashes':['parent'],'split':'test','speaker':'b','dataset':'X'}
    with pytest.raises(ValueError,match='Parent-source'):validate_lineage([a,b])
    b.update(source_hashes=[],speaker='a')
    with pytest.raises(ValueError,match='Speaker'):validate_lineage([a,b])

def test_real_checkpoint_polarity_and_compiled_parity():
    from voiceguard.vendor.aasist import Model
    e=Engine();x=decode(next(Path('data/demo').glob('*.wav')));features,score,logits=e.features(x)
    assert features.shape==(248,) and np.isfinite(features).all()
    assert score==pytest.approx(logits[0]-logits[1],abs=1e-5) # official class 0 spoof, 1 bonafide
    model=Model(json.loads((MODEL_ROOT/'config.json').read_text())['model_config']);model.load_state_dict(torch.load(MODEL_ROOT/'weights.pth',weights_only=True,map_location='cpu'));model.eval()
    with torch.inference_mode():_,eager=model(torch.from_numpy(fit_length(x).copy()).unsqueeze(0))
    np.testing.assert_allclose(logits,eager[0].numpy(),atol=1e-4,rtol=1e-4)
    silent=e.score(np.zeros(32000),'baseline');assert silent['verdict']=='Inconclusive'

def test_timestamp_packet_validation():
    from voiceguard.server import parse_packet
    x=np.zeros(HOP,dtype='<f4');p=struct.pack('<Id',1,8000)+x.tobytes();assert len(parse_packet(p,1))==HOP
    with pytest.raises(ValueError,match='sequence'):parse_packet(p,0)
    with pytest.raises(ValueError):parse_packet(b'bad',0)
    x[0]=np.nan
    with pytest.raises(ValueError,match='Invalid PCM'):parse_packet(struct.pack('<Id',0,0)+x.tobytes(),0)

def test_live_real_model_and_api_errors():
    from voiceguard.server import app,engine
    with TestClient(app) as client:
        assert client.post('/api/analyze?model=baseline',content=b'broken').status_code==422
        assert client.post('/api/analyze?model=baseline',content=wav(),headers={'origin':'https://untrusted.example'}).status_code==403
        x=decode(wav())
        with client.websocket_connect('/api/stream?model=baseline',headers={'origin':'http://127.0.0.1:8765'}) as ws:
            assert ws.receive_json()['type']=='ready'
            for i in range(4):
                ws.send_bytes(struct.pack('<Id',i,i*HOP)+x[i*HOP:(i+1)*HOP].astype('<f4').tobytes());result=ws.receive_json()
                assert result['end']==(i+1)*.5
                if i<3:assert result['score'] is None
            expected=engine.score(x,'baseline');assert result['score']==pytest.approx(expected['score'],abs=1e-5)
            ws.send_bytes(b'bad');assert ws.receive_json()['type']=='error'

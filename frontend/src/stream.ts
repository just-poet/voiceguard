import type {Bin} from './types';
export class StreamSession {
  private ws:WebSocket|null=null;private queue:ArrayBuffer[]=[];private inflight=false;private sequence=0;private sent=0;private stopped=false;
  private context:AudioContext|null=null;private media:MediaStream|null=null;private node:AudioWorkletNode|null=null;private timer:number|undefined;private staleTimer:number|undefined;private lastUpdate=performance.now();
  constructor(private model:string,private onBin:(b:Bin)=>void,private onState:(s:string)=>void,private onError:(s:string)=>void){}
  async connect(){
    await new Promise<void>((resolve,reject)=>{
      const ws=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/api/stream?model=${this.model}`);this.ws=ws;
      const timeout=window.setTimeout(()=>{reject(Error('Engine connection timed out'));this.stop();},10000);
      ws.onerror=()=>{clearTimeout(timeout);reject(Error('Cannot connect. The engine may be busy.'));};
      ws.onclose=()=>{clearTimeout(timeout);reject(Error('Stream closed'));if(!this.stopped)this.fail('Stream disconnected. Start a new session.');};
      ws.onmessage=event=>{
        const message=JSON.parse(event.data);
        if(message.type==='ready'){clearTimeout(timeout);this.onState('Warm-up · collecting 2 seconds');resolve();return;}
        if(message.type==='error'){this.fail(message.message);return;}
        if(message.type==='end'){this.stop();this.onState('Session limit reached');return;}
        if(message.type==='score'){
          this.inflight=false;this.lastUpdate=performance.now();message.roundtrip_ms=this.lastUpdate-this.sent;
          this.onBin(message);this.onState(message.state==='warmup'?'Warm-up · collecting 2 seconds':message.overloaded?'Overloaded · inference exceeded 500 ms':'Monitoring · latest past-only window');this.flush();
        }
      };
    });
    if(!this.stopped)this.staleTimer=window.setInterval(()=>{if(performance.now()-this.lastUpdate>1800)this.onState('Stale · waiting for an update');},300);
  }
  enqueue(chunk:ArrayBuffer){
    if(this.stopped)return;
    if(this.queue.length>=2){this.fail('Overloaded: two queued chunks. Session stopped without silently dropping audio.');return;}
    this.queue.push(chunk);this.flush();
  }
  private flush(){
    if(this.inflight||!this.queue.length||this.ws?.readyState!==WebSocket.OPEN)return;
    const data=this.queue.shift()!;const packet=new ArrayBuffer(12+data.byteLength);const header=new DataView(packet);header.setUint32(0,this.sequence,true);header.setFloat64(4,this.sequence*8000,true);new Uint8Array(packet,12).set(new Uint8Array(data));this.sequence++;this.inflight=true;this.sent=performance.now();this.ws.send(packet);
  }
  async microphone(){
    const media=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:false}});
    if(this.stopped){media.getTracks().forEach(t=>t.stop());return;}
    this.media=media;const context=new AudioContext();this.context=context;await context.audioWorklet.addModule('/processor.js');
    if(this.stopped){await context.close();return;}
    this.node=new AudioWorkletNode(context,'voiceguard-processor');this.node.port.onmessage=e=>this.enqueue(e.data);const source=context.createMediaStreamSource(media);source.connect(this.node);const mute=context.createGain();mute.gain.value=0;this.node.connect(mute).connect(context.destination);await context.resume();
  }
  replay(pcm:Float32Array,onComplete:()=>void){
    let offset=0;const start=performance.now();const tick=()=>{
      if(this.stopped)return;
      if(offset>=pcm.length){
        if(this.inflight||this.queue.length){this.timer=window.setTimeout(tick,30);return;}
        this.stop();this.onState('Replay complete');onComplete();return;
      }
      const chunk=new Float32Array(8000);chunk.set(pcm.subarray(offset,offset+8000));this.enqueue(chunk.buffer);offset+=8000;
      this.timer=window.setTimeout(tick,Math.max(0,start+(offset/16000+.5)*1000-performance.now()));
    };
    this.timer=window.setTimeout(tick,500);
  }
  private fail(message:string){this.stop();this.onError(message);this.onState('Stopped · needs attention');}
  stop(){this.stopped=true;clearTimeout(this.timer);clearInterval(this.staleTimer);this.queue=[];this.node?.disconnect();this.media?.getTracks().forEach(t=>t.stop());if(this.context?.state!=='closed')void this.context?.close();this.ws?.close();}
}

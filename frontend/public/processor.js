// Causal FIR low-pass + phase-accumulator resampling to 16 kHz.
class VoiceGuardProcessor extends AudioWorkletProcessor {
  constructor() {
    super();this.ring=new Float32Array(48);this.cursor=0;this.phase=0;this.last=0;this.chunk=new Float32Array(8000);this.position=0;
    this.taps=new Float64Array(48);const fc=Math.min(.45,7200/sampleRate);let sum=0;
    for(let i=0;i<48;i++){const n=i-23.5;const w=.54-.46*Math.cos(2*Math.PI*i/47);this.taps[i]=Math.sin(2*Math.PI*fc*n)/(Math.PI*n)*w;sum+=this.taps[i];}
    for(let i=0;i<48;i++)this.taps[i]/=sum;
  }
  process(inputs){
    const channels=inputs[0];if(!channels?.length)return true;
    for(let i=0;i<channels[0].length;i++){
      let value=0;for(const channel of channels)value+=channel[i]/channels.length;
      this.ring[this.cursor]=value;let filtered=value;
      if(sampleRate!==16000){filtered=0;for(let k=0;k<48;k++)filtered+=this.taps[k]*this.ring[(this.cursor-k+48)%48];}
      this.cursor=(this.cursor+1)%48;this.phase+=16000;
      if(this.phase>=sampleRate){
        this.phase-=sampleRate;const fraction=this.phase/16000;
        this.chunk[this.position++]=Math.max(-1,Math.min(1,filtered*(1-fraction)+this.last*fraction));
        if(this.position===8000){this.port.postMessage(this.chunk.buffer,[this.chunk.buffer]);this.chunk=new Float32Array(8000);this.position=0;}
      }
      this.last=filtered;
    }
    return true;
  }
}
registerProcessor('voiceguard-processor',VoiceGuardProcessor);

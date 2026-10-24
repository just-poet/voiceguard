const fs=require('node:fs');const vm=require('node:vm');const assert=require('node:assert/strict');const {test}=require('node:test');
for(const sampleRate of [16000,44100,48000])test(`Causal microphone resampler emits exact 500ms packets at ${sampleRate} Hz`,()=>{
 const packets=[];let Processor;
 const scope={sampleRate,Float32Array,Float64Array,Math,AudioWorkletProcessor:class{constructor(){this.port={postMessage:b=>packets.push(new Float32Array(b))};}},registerProcessor:(name,p)=>Processor=p};
 vm.runInNewContext(fs.readFileSync('frontend/public/processor.js','utf8'),scope);const processor=new Processor();
 for(let start=0;start<sampleRate*2;start+=128){const n=Math.min(128,sampleRate*2-start);processor.process([[new Float32Array(n).fill(.2),new Float32Array(n).fill(.4)]]);}
 assert.equal(packets.length,4);for(const packet of packets)assert.equal(packet.length,8000);
 assert.ok(Math.abs(packets[3][7999]-.3)<1e-5);
});

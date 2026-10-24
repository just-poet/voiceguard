// Transport-interface mocks only; real inference is tested by Python/WebSocket runs.
import test from 'node:test';import assert from 'node:assert/strict';import {StreamSession} from '../tmp/stream-test/stream.js';
globalThis.window=globalThis;globalThis.location={protocol:'http:',host:'127.0.0.1:8765'};
class FakeSocket{
 static OPEN=1;static instances=[];readyState=1;sent=[];closed=false;
 constructor(){FakeSocket.instances.push(this);setTimeout(()=>this.onmessage?.({data:JSON.stringify({type:'ready'})}),0);}
 send(data){this.sent.push(data);}
 close(){this.closed=true;this.readyState=3;this.onclose?.();}
 ack(){this.onmessage({data:JSON.stringify({type:'score',state:'warmup',score:null})});}
}
globalThis.WebSocket=FakeSocket;
test('Bounded transport stops rather than discarding queued audio',async()=>{
 const errors=[];const s=new StreamSession('baseline',()=>{},()=>{},e=>errors.push(e));await s.connect();
 for(let i=0;i<4;i++)s.enqueue(new ArrayBuffer(32000));
 assert.equal(errors.length,1);assert.match(errors[0],/Overloaded/);assert.ok(FakeSocket.instances.at(-1).closed);s.stop();
});
test('Transport attaches monotonic sample timestamps and waits for acknowledgments',async()=>{
 const s=new StreamSession('baseline',()=>{},()=>{},()=>{});await s.connect();const ws=FakeSocket.instances.at(-1);
 s.enqueue(new ArrayBuffer(32000));s.enqueue(new ArrayBuffer(32000));assert.equal(ws.sent.length,1);ws.ack();assert.equal(ws.sent.length,2);
 const first=new DataView(ws.sent[0]),second=new DataView(ws.sent[1]);assert.equal(first.getUint32(0,true),0);assert.equal(first.getFloat64(4,true),0);assert.equal(second.getUint32(0,true),1);assert.equal(second.getFloat64(4,true),8000);s.stop();
});

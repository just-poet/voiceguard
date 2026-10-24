const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

test('AudioWorklet preserves PCM packet boundaries and downmixes stereo', () => {
  const packets = [];
  let Processor;
  class AudioWorkletProcessor {
    constructor() { this.port = { postMessage: buffer => packets.push(new Float32Array(buffer)) }; }
  }
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../web/processor.js'), 'utf8'), {
    AudioWorkletProcessor, Float32Array, Math,
    registerProcessor: (name, constructor) => { assert.equal(name, 'chunk-processor'); Processor = constructor; }
  });
  const processor = new Processor();
  const left = new Float32Array(128).fill(.25), right = new Float32Array(128).fill(.75);
  for (let i = 0; i < 125; i++) assert.equal(processor.process([[left, right]]), true);
  assert.equal(packets.length, 2);
  for (const packet of packets) {
    assert.equal(packet.length, 8000);
    assert.ok([...packet].every(value => value === .5));
  }
  assert.equal(processor.position, 0);
});

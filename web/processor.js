class ChunkProcessor extends AudioWorkletProcessor {
  constructor() { super(); this.chunk = new Float32Array(8000); this.position = 0; }
  process(inputs) {
    const channels = inputs[0];
    if (!channels || !channels.length) return true;
    const frames = channels[0].length;
    for (let i = 0; i < frames; i++) {
      let sample = 0;
      for (let c = 0; c < channels.length; c++) sample += channels[c][i] / channels.length;
      this.chunk[this.position++] = Math.max(-1, Math.min(1, sample));
      if (this.position === 8000) {
        this.port.postMessage(this.chunk.buffer, [this.chunk.buffer]);
        this.chunk = new Float32Array(8000); this.position = 0;
      }
    }
    return true;
  }
}
registerProcessor('chunk-processor', ChunkProcessor);

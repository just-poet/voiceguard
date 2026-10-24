const $ = (selector) => document.querySelector(selector);
let file = null, result = null, audioURL = null, live = null, modelReady = false, busy = false, toastTimer, currentBlob = null, micRequest = 0;
const descriptions = {
  untrained: ['Model not trained', 'Signal inspection is available. Train on labeled speech to enable authenticity evidence.'],
  'insufficient-audio': ['More speech needed', 'Audio is too short, too quiet, or severely clipped for useful evidence.'],
  'warming-up': ['Building context…', 'The detector needs the first 2 seconds of audio before reporting evidence.'],
  uncertain: ['The evidence is uncertain.', 'Signal quality, channel disagreement, or a score near the threshold prevents a clear result.'],
  'synthetic-evidence': ['Synthetic speech evidence', 'The trained baseline found synthetic-like cues. This result needs verification and does not establish fraud.'],
  'genuine-evidence': ['Genuine speech evidence', 'The trained baseline found genuine-like cues. This cannot establish a speaker’s identity or make a call safe.']
};
function toast(message) { $('#toast').textContent = message; $('#toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => $('#toast').hidden = true, 7500); }
function page(name) { document.querySelectorAll('.page').forEach(e => e.hidden = e.id !== name); document.querySelectorAll('.nav').forEach(e => e.classList.toggle('active', e.dataset.page === name)); $('#breadcrumb').textContent = name === 'protocol' ? 'Research protocol' : 'Audio workbench'; }
document.querySelectorAll('[data-page]').forEach(e => e.onclick = () => page(e.dataset.page));
$('#setup-link').onclick = () => page('protocol');
function setModel(model) {
  modelReady = model.ready;
  const languages=model.training?.languages?.join(' / ') || 'speech';
  $('#model-banner strong').textContent = model.ready ? (model.training?.exploratory?'Exploratory':'Research')+' model loaded · '+languages : 'Model not trained · signal inspection available';
  $('#model-banner p').textContent = model.ready ? (model.training?.exploratory?'Public-data baseline. Speaker and unseen-tool holdouts are unverified; check the measured results before interpreting scores.':'Research model loaded. Review held-out results before interpreting its evidence.') : 'Add labeled genuine and synthetic speech, train locally, then restart the engine. No paid API or GPU required.';
  if(model.ready && !result)$('#verdict-copy').textContent='Upload or choose a held-out clip to inspect signal quality, local evidence and phone-channel consistency.';
}
fetch('/api/status').then(r => { if (!r.ok) throw Error('Cannot reach the local engine.'); return r.json(); }).then(s => { setModel(s.model); $('#connection').textContent = 'online'; }).catch(e => { $('#connection').textContent = 'offline'; toast(e.message); });
function selectTab(isLive) { micRequest++; if (live) stopLive(); $('#upload-pane').hidden = isLive; $('#live-pane').hidden = !isLive; $('#upload-tab').classList.toggle('selected', !isLive); $('#live-tab').classList.toggle('selected', isLive); $('#upload-tab').setAttribute('aria-selected', String(!isLive)); $('#live-tab').setAttribute('aria-selected', String(isLive)); }
$('#upload-tab').onclick = () => selectTab(false); $('#live-tab').onclick = () => selectTab(true);
function chooseFile(value) { if (busy) return; if (!value) return; if (value.size > 8 * 1024 * 1024) return toast('Choose a file under 8 MB.'); file = value; $('#filename').textContent = value.name; $('#analyze').disabled = false; }
$('#file-input').onchange = e => chooseFile(e.target.files[0]);
$('#dropzone').onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('#file-input').click(); } };
for (const event of ['dragenter', 'dragover']) $('#dropzone').addEventListener(event, e => { e.preventDefault(); $('#dropzone').classList.add('drag'); });
for (const event of ['dragleave', 'drop']) $('#dropzone').addEventListener(event, e => { e.preventDefault(); $('#dropzone').classList.remove('drag'); });
$('#dropzone').addEventListener('drop', e => chooseFile(e.dataTransfer.files[0]));
function clearPlayer() { if (audioURL) URL.revokeObjectURL(audioURL); audioURL = null; $('#player').hidden = true; $('#player').removeAttribute('src'); }
async function analyzeBlob(blob, diagnostic = false, metadata = null) {
  if (busy) return;
  busy = true; $('#analyze').disabled = true; $('#analyze').textContent = 'Inspecting audio…';
  document.querySelectorAll('[data-sample]').forEach(b => b.disabled = true);
  try {
    if (live) await stopLive();
    const response = await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: blob });
    const data = await response.json(); if (!response.ok) throw Error(data.detail || 'Analysis failed.');
    result = { ...data, source: diagnostic ? 'diagnostic-tone-not-speech' : metadata ? 'public-held-out-example' : 'uploaded-clip', example:metadata, exported_at: new Date().toISOString() };
    if (diagnostic) { result.diagnostic = true; result.summary = {score: null, label: 'untrained'}; result.bins = result.bins.map(b => ({...b, score:null, context_score:null, views:{}, context_views:{}, disagreement:null, context_disagreement:null, label:'diagnostic'})); }
    currentBlob = diagnostic ? null : blob; $('#replay').disabled = diagnostic;
    $('#example-label').textContent = metadata ? metadata.language + ' · dataset label: ' + (metadata.label ? 'synthetic' : 'genuine') : diagnostic ? 'Generated tones · not a speech benchmark' : '';
    clearPlayer(); audioURL = URL.createObjectURL(blob); $('#player').src = audioURL; $('#player').hidden = false;
    render();
    if (diagnostic) toast('Diagnostic signal loaded. Tones test the pipeline; they cannot validate speech detection.');
  } catch (e) { toast(e.message); }
  finally { busy = false; $('#analyze').disabled = !file; $('#analyze').innerHTML = 'Analyze audio <span>↗</span>'; document.querySelectorAll('[data-sample]').forEach(b => b.disabled = false); }
}
$('#analyze').onclick = () => { if (file) analyzeBlob(file); };
document.querySelectorAll('[data-sample]').forEach(button => button.onclick = async () => { try { const response = await fetch('/api/diagnostic/' + button.dataset.sample); if (!response.ok) throw Error('Could not load the diagnostic.'); await analyzeBlob(await response.blob(), true); } catch(e) { toast(e.message); } });
function renderSummary(summary, bin, diagnostic = false, scope = null) {
  const [title, copy] = descriptions[summary.label] || descriptions.uncertain;
  $('#verdict-title').textContent = diagnostic ? 'Signal check complete.' : title;
  $('#verdict-copy').textContent = diagnostic ? 'This is a generated tone, not speech. The waveform and signal quality checks work; authenticity scores are suppressed.' : copy;
  $('#verdict-kicker').textContent = diagnostic ? 'DIAGNOSTIC · NOT SPEECH' : scope || (result?.source === 'realtime-replay' ? 'REPLAY · LATEST 500 MS' : result?.source === 'live-microphone' ? 'MICROPHONE · LATEST 500 MS' : 'WHOLE-CLIP EVIDENCE');
  $('#score').textContent = summary.score == null ? '—' : summary.score.toFixed(3);
  $('#quality-level').textContent = (bin?.quality?.rms_dbfs ?? '—') + ' dBFS';
  const spread=summary.disagreement ?? bin?.disagreement;
  $('#disagreement').textContent = diagnostic || spread == null ? '—' : spread.toFixed(3);
  $('#latency').textContent = (bin?.processing_ms ?? '—') + ' ms';
  const views=summary.views || bin?.views || {};
  for(const name of ['native','mulaw','alaw']) {
    const value=diagnostic?null:views[name];$('#view-'+name).textContent=value==null?'—':value.toFixed(3);
    $('#bar-'+name).style.width=(value==null?0:value*100)+'%';
  }
}
function render() {
  if (!result) return;
  $('#canvas-empty').hidden = true; $('#duration').textContent = result.duration.toFixed(1) + ' s';
  $('#export').disabled = false; $('#mode-tag').textContent = result.diagnostic ? 'DIAGNOSTIC' : live ? (live.kind === 'replay' ? 'STREAM REPLAY' : 'LIVE SESSION') : result.source === 'realtime-replay' ? 'REPLAY COMPLETE' : result.source === 'live-microphone' ? 'SESSION ENDED' : result.duration.toFixed(1) + ' S CLIP';
  const last = result.bins.at(-1);
  renderSummary(result.summary, last ? {...last,quality:result.quality || last.quality} : null, result.diagnostic);
  const container = $('#segments'); container.replaceChildren();
  for (const bin of result.bins.slice(-60)) {
    const button = document.createElement('button'); button.className = 'segment ' + (result.diagnostic ? '' : bin.label);
    button.textContent = bin.start.toFixed(1) + '–' + bin.end.toFixed(1) + 's';
    const value = document.createElement('span'); value.textContent = bin.label === 'warming-up' ? 'Warm-up' : result.diagnostic || bin.score == null ? (bin.quality.usable ? 'Signal OK' : 'Low quality') : bin.score.toFixed(3); button.append(value);
    button.title = result.diagnostic ? 'Signal diagnostics only' : bin.label;
    button.onclick = () => { if (!$('#player').hidden) { $('#player').currentTime = bin.start; $('#player').play().catch(() => {}); } renderSummary({ score: result.diagnostic ? null : bin.score, label: bin.label }, bin, result.diagnostic, 'SELECTED SEGMENT EVIDENCE'); };
    container.append(button);
  }
  draw();
}
function draw() {
  const canvas = $('#timeline'), ctx = canvas.getContext('2d'); const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth; canvas.width = width * ratio; canvas.height = 180 * ratio; ctx.scale(ratio, ratio); ctx.clearRect(0, 0, width, 180);
  if (!result || !width) return;
  const duration = result.duration || .5;
  for (const b of result.bins) { const colors = {'synthetic-evidence': '#cf6a4525', 'genuine-evidence': '#a2bc7125', uncertain:'#c7b65a25'}; ctx.fillStyle = result.diagnostic ? '#dce6d225' : colors[b.label] || '#dce6d225'; ctx.fillRect(b.start / duration * width, 0, (b.end - b.start) / duration * width, 180); }
  const wave = result.waveform || [];
  const peak = Math.max(...wave, .01);
  ctx.fillStyle = '#6d8c78'; const count = wave.length;
  for (let i=0;i<count;i++) { const height = Math.max(2, wave[i] / peak * 78); ctx.fillRect(i / count * width, 90 - height/2, Math.max(1, width/count - 2), height); }
  if (!result.diagnostic && result.bins.some(b => b.score != null)) {
    ctx.beginPath(); ctx.strokeStyle = '#bc6241'; ctx.lineWidth = 2;
    let started=false; for (const b of result.bins) { if (b.score == null || b.label === 'warming-up') continue; const x=(b.start+b.end)/2/duration*width, y=160-b.score*140; if (!started) {ctx.moveTo(x,y);started=true;}else ctx.lineTo(x,y); } ctx.stroke();
    ctx.font='9px Segoe UI';ctx.fillStyle='#a06b50';ctx.fillText('Synthetic evidence score (0–1)',10,15);
  }
}
new ResizeObserver(draw).observe($('#timeline'));
$('#export').onclick = () => {
  if (!result) return;
  const blob = new Blob([JSON.stringify({...result, exported_at: new Date().toISOString()}, null, 2)], {type:'application/json'});
  const url=URL.createObjectURL(blob), a=document.createElement('a'); a.href=url;a.download='callguard-report.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
async function startLive() {
  if (busy || live) return;
  $('#record').disabled = true;
  let state = null;
  const requestId = ++micRequest;
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw Error('Microphone access requires localhost and browser permission.');
    const stream = await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:false,noiseSuppression:false,autoGainControl:false},video:false});
    if (requestId !== micRequest) { stream.getTracks().forEach(t => t.stop()); return; }
    state = {stream, stopped:false, pending:false, kind:'microphone'}; live = state;
    state.context = new AudioContext({sampleRate:16000});
    if (state.context.sampleRate !== 16000) throw Error('This browser cannot provide 16 kHz audio. Upload a clip instead.');
    await state.context.audioWorklet.addModule('/static/processor.js');
    await state.context.resume();
    state.socket = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/api/stream');
    await new Promise((resolve,reject) => {state.socket.onopen=resolve;state.socket.onerror=()=>reject(Error('Streaming connection failed.'));});
    clearPlayer();result={duration:0,summary:{score:null,label:modelReady?'warming-up':'untrained'},bins:[],waveform:[],source:'live-microphone',model:{ready:modelReady},resolution_seconds:.5};
    state.socket.onmessage = event => {
      const message = JSON.parse(event.data);
      if (message.type === 'score') {
        state.pending=false; result.bins.push(message);result.duration=message.end;result.summary={score:message.score,label:message.label};render();
      } else if (message.type === 'ready') { result.model = message.model; }
      else if (message.type === 'error') { toast(message.message);stopLive(); }
      else if (message.type === 'end') { toast(message.message);stopLive(); }
    };
    state.socket.onclose=()=>{if(!state.stopped){toast('Microphone session ended.');stopLive();}};
    state.source=state.context.createMediaStreamSource(stream);state.node=new AudioWorkletNode(state.context,'chunk-processor');
    state.node.port.onmessage=e=>{
      if(state.stopped)return;
      if (state.pending || state.socket.readyState !== WebSocket.OPEN) { toast('Processing fell behind the 500 ms cadence. Session stopped to preserve timestamps.');stopLive();return; }
      const samples=new Float32Array(e.data);let peak=0;for(const s of samples)peak=Math.max(peak,Math.abs(s));result.waveform.push(peak);
      state.pending=true;state.socket.send(e.data);
    };
    state.gain=state.context.createGain();state.gain.gain.value=0;
    state.source.connect(state.node);state.node.connect(state.gain);state.gain.connect(state.context.destination);
    $('#record').textContent='Stop microphone';$('#mic-title').textContent='Listening on this computer';$('#mic-orb').classList.add('recording');
    render();
  } catch(e) { await stopLive(); toast(e.name==='NotAllowedError'?'Microphone permission was denied. Allow it in the browser or upload a clip.':e.message); }
  finally { $('#record').disabled=false; }
}
async function stopLive() {
  const state = live; if(!state) return; state.stopped=true;live=null;
  state.node?.disconnect();state.source?.disconnect();state.stream?.getTracks().forEach(t=>t.stop());
  if(state.socket && state.socket.readyState < 2) state.socket.close();
  if(state.context && state.context.state !== 'closed') await state.context.close();
  $('#record').textContent='Start microphone';$('#mic-title').textContent='Listen in real time';$('#mic-orb').classList.remove('recording');
  $('#replay').textContent='Replay at call speed ↗'; $('#replay').disabled=!currentBlob;
  if(result){result.session_ended=true;render();}
}
$('#record').onclick=()=>live?stopLive():startLive();
window.addEventListener('pagehide',()=>{if(live){live.stream?.getTracks().forEach(t=>t.stop());live.socket?.close();}});

async function loadExamples() {
  const container = $('#public-examples');
  try {
    const response = await fetch('/api/examples'); if(!response.ok) throw Error('Examples not available yet.');
    const examples = await response.json(); container.replaceChildren();
    if(!examples.length) {container.textContent='Run the public dataset downloader to add examples.';return;}
    for(const example of examples) {
      const button=document.createElement('button');button.textContent=example.language+' · '+(example.label?'synthetic':'genuine')+' ↗';
      button.onclick=async()=>{if(busy)return;try{button.disabled=true;const r=await fetch('/api/examples/'+encodeURIComponent(example.id));if(!r.ok)throw Error('Example could not be loaded.');await analyzeBlob(await r.blob(),false,example);}catch(e){toast(e.message);}finally{button.disabled=false;}};
      container.append(button);
    }
  }catch(e){container.textContent=e.message;}
}
loadExamples();

async function loadBenchmark() {
  const container=$('#benchmark-results');
  try {
    const response=await fetch('/api/evaluation');if(!response.ok)throw Error('Evaluation is not available yet.');
    const report=await response.json();container.replaceChildren();
    const note=document.createElement('p');note.textContent=report.available?'Exploratory held-out subset. These are measured results, not a guarantee on real phone calls.':'Run the evaluator to populate this table. No results have been assumed.';container.append(note);
    if(!report.available)return;
    const table=document.createElement('table');table.innerHTML='<thead><tr><th>Condition</th><th>EER ↓</th><th>Segment F1 ↑</th><th>Accepted</th><th>p95 / bin</th></tr></thead>';
    const body=document.createElement('tbody');
    for(const [name,row] of Object.entries(report.conditions)) {
      const tr=document.createElement('tr');
      for(const value of [name.replaceAll('_',' '),row.clip_eer?row.clip_eer.eer_percent.toFixed(2)+'%':'—',row.localization?row.localization.f1.toFixed(3):'—',(row.accepted_fraction*100).toFixed(1)+'%',row.bin_processing_p95_ms.toFixed(1)+' ms']) {const td=document.createElement('td');td.textContent=value;tr.append(td);}body.append(tr);
    }
    table.append(body);container.append(table);
    const languages=document.createElement('p');languages.textContent='Clean EER by language: '+Object.entries(report.conditions.clean.by_language).map(([name,m])=>name+' '+(m.eer?m.eer.eer_percent.toFixed(2)+'%':'unavailable')+' (n='+m.clips+')').join(' · ');container.append(languages);
    const limitations=document.createElement('p');limitations.className='benchmark-warning';limitations.textContent='Phone/noise EER target: '+(report.phase2.within_2_5x_clean?'met on this subset':'NOT MET')+' · Speaker/tool holdout: '+(report.phase2.speaker_and_generator_holdout_verified?'verified':'unverified')+' · Phase 1 retention: '+(report.phase2.within_2_phase1_points==null?'baseline unavailable':report.phase2.within_2_phase1_points?'within target':'outside target')+' · Partial clips: '+report.phase2.partial_clips;container.append(limitations);
  }catch(e){container.textContent=e.message;}
}
$('#refresh-results').onclick=loadBenchmark;loadBenchmark();

async function startReplay() {
  if(busy || live || !currentBlob)return;
  const blob=currentBlob, previousExample=result?.example || null;
  const state={kind:'replay',stopped:false};live=state;
  $('#replay').textContent='Stop replay ■';
  try {
    state.context=new AudioContext({sampleRate:16000});
    const decoded=await state.context.decodeAudioData(await blob.arrayBuffer());
    if(decoded.sampleRate!==16000)throw Error('Replay requires a browser supporting a 16 kHz audio context.');
    const mono=new Float32Array(decoded.length);
    for(let c=0;c<decoded.numberOfChannels;c++){const values=decoded.getChannelData(c);for(let i=0;i<mono.length;i++)mono[i]+=values[i]/decoded.numberOfChannels;}
    await state.context.close();
    if(state.stopped)return;
    state.socket=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/api/stream');
    await new Promise((resolve,reject)=>{state.socket.onopen=resolve;state.socket.onerror=()=>reject(Error('Replay connection failed.'));});
    result={duration:0,summary:{score:null,label:modelReady?'warming-up':'untrained'},bins:[],waveform:[],source:'realtime-replay',example:previousExample,resolution_seconds:.5,model:{ready:modelReady},roundtrip_ms:[],source_duration_seconds:mono.length/16000,padded_tail_seconds:(Math.ceil(mono.length/8000)*8000-mono.length)/16000};
    let resolveResponse=null,rejectResponse=null;
    state.socket.onmessage=e=>{const msg=JSON.parse(e.data);if(msg.type==='ready'){result.model=msg.model;}else if(msg.type==='score'){result.bins.push(msg);result.duration=msg.end;result.summary={score:msg.score,label:msg.label};render();resolveResponse?.();}else if(msg.type==='error'){rejectResponse?.(Error(msg.message));}};
    state.socket.onerror=()=>rejectResponse?.(Error('Replay connection failed.'));
    state.socket.onclose=()=>rejectResponse?.(Error('Replay connection closed.'));
    const started=performance.now();
    for(let offset=0;offset<mono.length && !state.stopped;offset+=8000){
      const deadline=started+(offset/16000+.5)*1000;
      await new Promise(r=>setTimeout(r,Math.max(0,deadline-performance.now())));
      if(state.stopped)break;
      const chunk=new Float32Array(8000);chunk.set(mono.subarray(offset,offset+8000));let peak=0;for(const s of chunk)peak=Math.max(peak,Math.abs(s));result.waveform.push(peak);
      const sent=performance.now();
      await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('No streaming response within 5 seconds.')),5000);resolveResponse=()=>{clearTimeout(timer);resolve();};rejectResponse=e=>{clearTimeout(timer);reject(e);};state.socket.send(chunk.buffer);});
      result.roundtrip_ms.push(Math.round((performance.now()-sent)*100)/100);
    }
    if(!state.stopped)toast('Streaming replay complete. Export includes per-update round-trip timings.');
  }catch(e){if(!state.stopped)toast(e.message);}
  finally{if(live===state)await stopLive();}
}
$('#replay').onclick=()=>live?.kind==='replay'?stopLive():startReplay();

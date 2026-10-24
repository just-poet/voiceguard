"""Build the measured report and a portable demonstration ZIP."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports'
INK='#173c38';MUTED='#64756d';LIME='#d8ee80';ORANGE='#c56f4b'


def load(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))


def main():
    report=load('reports/evaluation.json');stream=load('reports/streaming-verification.json')
    model=load('models/detector.json');provenance=load('data/indictts/PROVENANCE.json')
    phase=report['phase2'];clean=report['conditions']['clean']
    figure_dir=REPORTS/'figures';figure_dir.mkdir(exist_ok=True)
    names=list(report['conditions'])
    labels=['Clean','8 kHz','mu-law','A-law','8k + noise','mu + noise','A + noise']
    eers=[report['conditions'][k]['clip_eer']['eer_percent'] for k in names]
    native=[report['conditions'][k]['native_only_ablation_eer']['eer_percent'] for k in names]
    local_f1=[report['conditions'][k]['localization']['f1'] for k in names]
    long_f1=[report['conditions'][k]['localization_long_head_ablation']['f1'] for k in names]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'axes.titleweight':'bold'})
    fig,axes=plt.subplots(2,1,figsize=(9,6),constrained_layout=True)
    xs=np.arange(len(names));width=.34
    axes[0].bar(xs-width/2,eers,width,color=INK,label='Selected channel weighting')
    axes[0].bar(xs+width/2,native,width,color='#a6b98d',label='Native-only ablation')
    axes[0].set_ylabel('EER (%) - lower is better');axes[0].set_xticks(xs,labels);axes[0].legend(frameon=False,fontsize=8)
    axes[0].set_title('Full-clip detection: 101 held-out clips',loc='left',pad=10)
    axes[1].plot(xs,local_f1,'o-',color=INK,label='500 ms local head',linewidth=2)
    axes[1].plot(xs,long_f1,'s--',color=ORANGE,label='2 second context head',linewidth=1.5)
    axes[1].set_ylabel('Segment F1 - higher is better');axes[1].set_ylim(0,1);axes[1].set_xticks(xs,labels);axes[1].legend(frameon=False,fontsize=8)
    axes[1].set_title('Localization: 100 annotated partial clips',loc='left',pad=10)
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(figure_dir/'evaluation.png',dpi=180,facecolor='white');plt.close(fig)

    headers=['Condition','EER (%)','Native EER (%)','Local F1','Long F1','Accepted (%)','p95 ms/bin']
    table=[headers]
    for name in names:
        r=report['conditions'][name]
        table.append([name,f"{r['clip_eer']['eer_percent']:.2f}",f"{r['native_only_ablation_eer']['eer_percent']:.2f}",f"{r['localization']['f1']:.3f}",f"{r['localization_long_head_ablation']['f1']:.3f}",f"{r['accepted_fraction']*100:.1f}",f"{r['bin_processing_p95_ms']:.1f}"])
    with (REPORTS/'metrics.csv').open('w',newline='',encoding='utf-8') as f:csv.writer(f).writerows(table)
    with (REPORTS/'trial-scores.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['condition','file','language','label','partial','score','decision','sha256'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:t[k] for k in fields} for t in report['trials'])

    source_rows=[json.loads(line) for line in (ROOT/'data/indictts/manifest.jsonl').read_text().splitlines()]
    counts=[]
    for language in ('English','Hindi','Tamil'):
        counts.append([language]+[sum(r['language']==language and r['split']==split for r in source_rows) for split in ('train','dev','test')])
    selection=load('reports/robust-selection.json') if (REPORTS/'robust-selection.json').exists() else load('reports/model-selection.json')
    model_hash=hashlib.sha256((ROOT/'models/detector.json').read_bytes()).hexdigest()
    robustness='MET on this subset' if phase['within_2_5x_clean'] else 'NOT MET'
    method=model.get('kind','linear');weights=model.get('view_weights',{'native':1/3,'mulaw':1/3,'alaw':1/3})
    markdown=['# CallGuard - measured submission report','',
              'Local CPU speech-authenticity prototype for English, Hindi and Tamil. Prepared 24 October 2026.','',
              f"**Clean EER: {phase['clean_eer_percent']:.2f}%. Worst degraded EER: {phase['worst_degraded_eer_percent']:.2f}%.**",
              f"The 2.5x clean-EER target is **{robustness}**. These results come from a small exploratory subset; speaker and generator holdouts cannot be verified.",'',
              f"**Streaming:** 500 ms updates; {stream['roundtrip_p95_ms']:.1f} ms p95 local round-trip latency; maximum offline/stream score difference {stream['max_score_difference']:.5f}.",'',
              '## Data and selection','',
              '720 public labeled clips, 120 per class per language. Normalized text groups determine the train/development/test split. The official challenge test partition was not used because its labels are unavailable.','',
              '| Language | Train | Development | Test |','|---|---:|---:|---:|']
    markdown += ['| '+' | '.join(map(str,row))+' |' for row in counts]
    markdown += ['',f"Selected classifier: **{method}**. Channel weights: `{weights}`. Both window heads use separate development-selected thresholds.",'',
                 'The initial test run exposed weak phone/noise robustness. Its report is retained as `initial-evaluation.json`. The bounded follow-up model/weight selection used development data and a clean-EER guardrail; its complete record is in `robust-selection.json`.','',
                 '## Measured results','', '| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]
    markdown += ['| '+' | '.join(row)+' |' for row in table[1:]]
    markdown += ['', '![Measured detection and localization](figures/evaluation.png)','',
                 'EER uses all full-clip scores. Acceptance coverage includes full and partial clips. Localization uses all partial-clip bins; bins ending before 2 seconds receive a negative prediction during warm-up. The selected native-only weighting makes its EER identical to the native-only ablation; codec probes still contribute uncertainty flags.','',
                 '## Requirement status','',
                 f"- Phone-channel degradation target: {robustness}; worst/clean ratio {phase['degradation_ratio']:.2f}x." if phase['degradation_ratio'] is not None else '- Ratio undefined because clean EER is zero; see the absolute inequality in JSON.',
                 '- 100 partial clips: generated, with timestamp labels and source hashes; 34 English, 33 Hindi, 33 Tamil.',
                 '- Same-speaker partials: unverified. Speaker changes and splice artifacts can confound localization.',
                 '- 500 ms streaming: verified with a timed local replay. Physical microphone/cellular capture were not verified.',
                 '- Unseen synthesis-tool generalization: unverified because the dataset lacks tool identities.',
                 '- Original Phase 1 retention: not assessable; no original model/results/test protocol was supplied.','',
                 '## Reproduce','', '```powershell',
                 'python tools/fetch_public_subset.py --per-class 120',
                 'python -m callguard.cli train data/indictts/manifest.jsonl --exploratory --classifier linear --output models/baseline-linear.json',
                 'python -m callguard.cli train data/indictts/manifest.jsonl --exploratory --classifier extra-trees --output models/candidate-trees.json',
                 'python -m callguard.cli train data/indictts/manifest.jsonl --exploratory --classifier rbf-svm --output models/candidate-svm.json',
                 'python tools/robust_select.py data/indictts/manifest.jsonl models/baseline-linear.json models/candidate-trees.json models/candidate-svm.json',
                 'python -m callguard.cli partials data/indictts/manifest.jsonl --count 100 --exploratory',
                 'python -m callguard.cli evaluate data/indictts/manifest.jsonl --partials data/partials/manifest.jsonl','```','',
                 '## Evidence and credits','',
                 f'Model SHA-256: `{model_hash}`.','',
                 'Data: [SherryT997 / IndicTTS Deepfake Detection Challenge](https://huggingface.co/datasets/SherryT997/IndicTTS-Deepfake-Challenge-Data), CC BY 4.0, revision `57347517658ae989597d8cef303cffb647ed2434`. Changes include subsampling, duration filtering/cropping, telephone transforms and annotated splicing.','',
                 'Prior work: [ASVspoof 2021](https://github.com/asvspoof-challenge/2021), [compression-aware detection](https://arxiv.org/abs/2501.05545), and [partial speech localization](https://arxiv.org/abs/2605.29531). The project makes no world-first novelty claim.','',
                 'A score is neither a calibrated fraud probability nor proof of the caller\'s identity. Real-world validation remains necessary.']
    (REPORTS/'SUBMISSION_REPORT.md').write_text('\n'.join(markdown),encoding='utf-8')

    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleCG',fontName='Helvetica',fontSize=34,leading=38,textColor=colors.HexColor(INK),spaceAfter=16))
    styles.add(ParagraphStyle(name='SubCG',fontName='Helvetica-Bold',fontSize=10,leading=14,textColor=colors.HexColor(MUTED),spaceAfter=14))
    styles.add(ParagraphStyle(name='HCG',fontName='Helvetica-Bold',fontSize=15,leading=19,textColor=colors.HexColor(INK),spaceBefore=13,spaceAfter=10))
    styles.add(ParagraphStyle(name='BodyCG',fontName='Helvetica',fontSize=10,leading=15,textColor=colors.HexColor(INK),spaceAfter=9))
    styles.add(ParagraphStyle(name='SmallCG',fontName='Helvetica',fontSize=8,leading=12,textColor=colors.HexColor(MUTED),spaceAfter=8))
    styles.add(ParagraphStyle(name='CodeCG',fontName='Courier',fontSize=8,leading=12,textColor=colors.HexColor(INK),backColor=colors.HexColor('#f0f4e8'),borderPadding=10,spaceBefore=8,spaceAfter=12))
    P=lambda text,style='BodyCG':Paragraph(text,styles[style])
    story=[]
    def title(kicker,title):story.extend([P(kicker.upper(),'SubCG'),P(title,'TitleCG')])
    def heading(text):story.append(P(text,'HCG'))
    def paragraph(text):story.append(P(text))
    def table_block(data,widths=None,small=False):
        rows=[[P(escape(str(value)),'SmallCG' if small else 'BodyCG') for value in row] for row in data]
        t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf0df')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#a5b394')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#dfe5d8'))]))
        story.append(t);story.append(Spacer(1,10))
    width=A4[0]-96

    title('CallGuard / Research edition / 24 October 2026','Listen beyond\nthe voice.'.replace('\n','<br/>'))
    paragraph('A free, local CPU prototype for short-clip speech authenticity and partial-fake localization in English, Hindi and Tamil.')
    story.append(Spacer(1,15))
    table_block([['CLEAN EER','WORST CHANNEL EER','STREAM p95'],[f"{phase['clean_eer_percent']:.2f}%",f"{phase['worst_degraded_eer_percent']:.2f}%",f"{stream['roundtrip_p95_ms']:.1f} ms"]],[width/3]*3)
    heading('What was built')
    paragraph('A local browser application with upload analysis, microphone streaming, call-speed replay, half-second evidence bins, channel comparison, uncertainty flags, and JSON export. Training, evaluation and source provenance are included.')
    paragraph('The detector uses a 500 ms localization head alongside a 2 second context head. Native and simulated G.711 channel views provide both scores and disagreement evidence. No paid inference service is used.')
    heading('The central result')
    paragraph(f"The phone/noise target is <b>{robustness}</b> on this subset. The clean-to-worst EER values are {phase['clean_eer_percent']:.2f}% and {phase['worst_degraded_eer_percent']:.2f}%. The initial model result is preserved for transparency.")
    paragraph('<b>Scope:</b> This is an exploratory public-data result. Speaker and synthesis-tool identities are unavailable, so unseen-speaker and unseen-tool generalization are not established. Original Phase 1 retention cannot be assessed without its baseline.')
    story.append(PageBreak())

    title('01 / Data and method','A reproducible<br/>small-data experiment.')
    paragraph('The prepared subset contains 720 labeled clips: 120 genuine and 120 synthetic examples per language. Clips shorter than 2 seconds are excluded; longer clips are cropped to 6 seconds. Identical normalized transcript text stays in one split.')
    table_block([['Language','Train','Development','Test']]+counts,[width*.4,width*.2,width*.2,width*.2])
    paragraph('The source is the labeled train split of the IndicTTS Deepfake Detection Challenge dataset, published by SherryT997 under CC BY 4.0. The official unlabelled test split was not used. Dataset revision: 57347517658ae989597d8cef303cffb647ed2434.')
    heading('Signal path')
    for text in ['1. Decode mono speech and resample to 16 kHz.',
                 '2. Compute 88 LFCC, temporal-delta and spectral-summary features.',
                 '3. Analyze native, mu-law and A-law views with short and long heads.',
                 '4. Apply development-selected channel weights and decision thresholds.',
                 '5. Use disagreement, low audio level, clipping and threshold proximity to abstain.']:
        paragraph(text)
    heading('Frozen model configuration')
    paragraph(f"Classifier: <b>{escape(method)}</b>. View weights (native / mu-law / A-law): {weights['native']:.3f} / {weights['mulaw']:.3f} / {weights['alaw']:.3f}. Clip threshold: {model['threshold']:.5f}; local-bin threshold: {model['local_threshold']:.5f}.")
    paragraph('Uncertainty uses a declared score margin of 0.08 and view-spread limit of 0.25. These are engineering heuristics, not calibrated confidence intervals.')
    story.append(PageBreak())

    title('02 / Evaluation','Phone channels expose<br/>the difficult cases.')
    paragraph('EER is computed on all 101 full-clip scores, including abstentions. Localization uses 100 partial clips. Degraded conditions apply a 300-3400 Hz bandpass at 8 kHz, optional G.711 coding, and optional 10 dB Gaussian noise.')
    pdf_table=[['Condition','EER %','Local F1','Accepted %','p95 ms']]
    for name in names:
        r=report['conditions'][name];pdf_table.append([name.replace('_noise10',' + noise'),f"{r['clip_eer']['eer_percent']:.2f}",f"{r['localization']['f1']:.3f}",f"{100*r['accepted_fraction']:.1f}",f"{r['bin_processing_p95_ms']:.1f}"])
    table_block(pdf_table,[width*.32,width*.16,width*.16,width*.18,width*.18],small=True)
    story.append(Image(str(figure_dir/'evaluation.png'),width=width*.90,height=width*.60))
    story.append(P('Accepted coverage includes both full and partial clips. A low EER does not measure fraud prevalence, identity verification, or generalization beyond this corpus.','SmallCG'))
    story.append(PageBreak())

    title('03 / Language and streaming','Inspect each language.<br/>Measure the real loop.')
    lang_table=[['Language','Full clips','Clean EER %','Clean segment F1']]
    for lang,data in clean['by_language'].items():lang_table.append([lang,data['clips'],f"{data['eer']['eer_percent']:.2f}",f"{data.get('localization',{}).get('f1',0):.3f}"])
    table_block(lang_table,[width*.25]*4)
    paragraph('The small test counts mean that individual-language rates are unstable. A zero observed error is not proof of perfect detection. Speaker identity is unknown and corpus/source artifacts may influence the results.')
    heading('Partial manipulation')
    paragraph('The tool generated 100 partial clips: 34 English, 33 Hindi and 33 Tamil. Each inserts a 0.5-2 second synthetic span after the warm-up, with 10 ms crossfades and source hashes. Same-language pairing is enforced, but same-speaker identity cannot be verified.')
    paragraph('A bin is synthetic when at least half its duration overlaps an inserted span. Bins ending before 2 seconds are counted as negative during warm-up. All bins remain in the F1 denominator. Splices and speaker changes may make the task easier than realistic cloned speech.')
    heading('Timed local WebSocket verification')
    table_block([['Measurement','Observed'],['Chunks / cadence','12 chunks / 500 ms'],['p50 round trip',f"{stream['roundtrip_p50_ms']:.2f} ms"],['p95 round trip',f"{stream['roundtrip_p95_ms']:.2f} ms"],['Median arrival interval',f"{stream['arrival_interval_p50_ms']:.2f} ms"],['Maximum offline/stream score difference',f"{stream['max_score_difference']:.5f}"]],[width*.65,width*.35],small=True)
    paragraph('This verifies the local browser/API design and timed PCM replay. Physical microphone hardware and cellular-call interception were not verified. The browser does not directly intercept phone calls.')
    story.append(PageBreak())

    title('04 / Submission and next work','Make the evidence<br/>easy to reproduce.')
    heading('Run the demonstration')
    paragraph('Double-click START_CALLGUARD.cmd, then open http://127.0.0.1:8765. First-time setup installs free Python dependencies. The prepared model and demo examples are included in the ZIP. After setup, analysis runs offline.')
    heading('Requirement ledger')
    table_block([['Requirement','Status'],['Two codecs, 8 kHz, added noise','Implemented and measured'],['Worst EER at most 2.5x clean',robustness],['100 partial fakes and 500 ms F1','Implemented; speaker matching unverified'],['Update every 500 ms','Verified by timed local replay'],['Unseen synthesis tools','Unverified: tool identities unavailable'],['Original Phase 1 within 2 points','Not assessable: baseline unavailable']],[width*.57,width*.43],small=True)
    heading('Contribution and limitations')
    paragraph('The project combines two time scales with counterfactual phone-channel evidence in an auditable CPU application. Compression robustness and partial-fake localization have prior art. No world-first claim is made. A rigorous next step requires identified, held-out generators and speakers, genuine edit controls, code-switched speech, and real telephony recordings.')
    heading('Sources and artifact integrity')
    for label,url in [('IndicTTS dataset and CC BY 4.0 credit','https://huggingface.co/datasets/SherryT997/IndicTTS-Deepfake-Challenge-Data'),('ASVspoof 2021 official baseline/evaluation','https://github.com/asvspoof-challenge/2021'),('Compression-aware detection research','https://arxiv.org/abs/2501.05545'),('Partial-fake localization and cross-corpus study','https://arxiv.org/abs/2605.29531')]:
        story.append(P(f'<link href="{url}" color="{INK}">{escape(label)}</link>','SmallCG'))
    story.append(P('Model SHA-256: '+model_hash,'SmallCG'))
    story.append(P('Complete trial scores, model selection, annotations and reproducible commands are in the accompanying JSON/CSV/Markdown files.','SmallCG'))

    output=ROOT/'output/pdf';output.mkdir(parents=True,exist_ok=True)
    pdf=output/'CallGuard_Submission_Report.pdf'
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#dfe5d8'));canvas.line(48,43,A4[0]-48,43)
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor(MUTED));canvas.drawString(48,29,'CALLGUARD  /  LOCAL CPU RESEARCH PROTOTYPE');canvas.drawRightString(A4[0]-48,29,f'{doc.page}');canvas.restoreState()
    SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=40,bottomMargin=59,title='CallGuard - Measured Submission Report',author='CallGuard project').build(story,onFirstPage=footer,onLaterPages=footer)

    demo=ROOT/'data/demo';demo.mkdir(parents=True,exist_ok=True);seen=set();demo_rows=[]
    for r in source_rows:
        key=(r['language'],r['label'])
        if r['split']=='test' and key not in seen:
            seen.add(key);source=ROOT/'data/indictts'/r['path'];shutil.copy2(source,demo/source.name)
            demo_rows.append({**r,'path':source.name})
    (demo/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in demo_rows),encoding='utf-8')
    (demo/'ATTRIBUTION.txt').write_text('Public examples from SherryT997 / IndicTTS Deepfake Detection Challenge. CC BY 4.0.\n'+provenance['dataset']+'\nChanges: subset selection, mono/16kHz conversion, duration cropping.\nNo correctness-based example selection.\n',encoding='utf-8')
    package=ROOT/'output/CallGuard_Submission.zip';files=[]
    for folder in ('callguard','web','tools','tests','docs'):
        files += [p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [ROOT/name for name in ('README.md','START_CALLGUARD.cmd','start.ps1','requirements.txt','requirements-lock.txt','requirements-dev.txt','requirements-report.txt','.gitignore')]
    files += [ROOT/'models/detector.json',pdf]
    files += [p for p in REPORTS.rglob('*') if p.is_file() and 'development-cache' not in p.parts]
    files += list(demo.glob('*'))
    files += list((ROOT/'data/partials').glob('*'))
    files += [ROOT/'data/indictts/PROVENANCE.json',ROOT/'data/indictts/SOURCE_README.md']
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(set(files)):archive.write(path,path.relative_to(ROOT))
    print(json.dumps({'pdf':str(pdf),'markdown':str(REPORTS/'SUBMISSION_REPORT.md'),'zip':str(package),'zip_mb':round(package.stat().st_size/1e6,2)},indent=2))


if __name__=='__main__':main()

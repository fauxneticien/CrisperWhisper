// CrisperWhisper 2.0 turbo in the browser: Transformers.js + ONNX Runtime Web.
//
// Long audio follows the Python package's default "continuation" strategy
// (crisperwhisper/longform/continuation.py): 30 s windows advancing by 26 s,
// each prompted with the last 12 confirmed words as "<ctx> ... <ectx>", and
// trailing words that start inside the 4 s overlap dropped (at most 2), since
// the next window hears them again. Not ported yet: early-EOT recovery,
// temperature fallback and Viterbi word timing. Word times come from
// Transformers.js' cross-attention DTW (no pauses); utterance start/end times
// are then trimmed to speech regions from Silero VAD (vad.js).

import {
  AutoProcessor, AutoTokenizer, WhisperForConditionalGeneration,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/dist/transformers.min.js';
import { buildEaf, downloadText, fileStem } from './eaf.js';
import { SileroVAD, detectSpeech, trimToSpeech } from './vad.js';

const MODEL_ID = 'Masterx/CrisperWhisper2.0-turbo-ONNX';
const SAMPLE_RATE = 16000;
const CHUNK_S = 30;        // LongformConfig.chunk_duration
const STRIDE_S = 26;       // LongformConfig.stride
const CONTEXT_WORDS = 12;  // LongformConfig.context_words
const DROP_WORDS = 2;      // LongformConfig.drop_words
const MAX_NEW_TOKENS = 256;
const MODE_TAGS = '[verbatim_1][verbatim_2][verbatim_3][verbatim_4][verbatim_5]';
const LANGUAGES = [
  ['fr', 'French'], ['en', 'English'], ['de', 'German'], ['es', 'Spanish'],
  ['it', 'Italian'], ['pt', 'Portuguese'], ['nl', 'Dutch'], ['pl', 'Polish'], ['sv', 'Swedish'], ['da', 'Danish'],
  ['no', 'Norwegian'], ['fi', 'Finnish'], ['cs', 'Czech'], ['ro', 'Romanian'], ['hu', 'Hungarian'], ['el', 'Greek'],
  ['ru', 'Russian'], ['uk', 'Ukrainian'], ['tr', 'Turkish'], ['ca', 'Catalan'],
];
// Prompt and control tokens that never belong in the transcript.
const CONTROL_TOKEN = /^(<\|[^|]*\|>|\[(verbatim|intended)_\d+\]|<\/?e?(ctx|htx|vtx)>)$/;

const $ = id => document.getElementById(id);
const state = { segments: [], fileName: '' };

// ---------- helpers ----------
function formatClock(seconds) {
  const m = Math.floor(seconds / 60);
  return `${m}:${(seconds - m * 60).toFixed(2).padStart(5, '0')}`;
}
function formatMB(bytes) { return `${(bytes / 1e6).toFixed(0)} MB`; }
function formatDuration(seconds) {
  const s = Math.max(0, Math.round(seconds));
  return s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, '0')} s`;
}

/** Transcription progress: windows done out of total, with a time estimate from the average window. */
function showProgress(done, total, startedMs) {
  $('tprogress').hidden = false;
  $('tprogress-bar').style.width = `${(100 * done / total).toFixed(1)}%`;
  $('tprogress-text').textContent = `Transcribing: ${done} of ${total} windows (${Math.round(100 * done / total)}%)`;
  const elapsed = (performance.now() - startedMs) / 1000;
  $('tprogress-eta').textContent = done > 0 && done < total
    ? `about ${formatDuration(elapsed / done * (total - done))} left`
    : done === 0 ? 'starting…' : '';
}

function progressRow(file) {
  let row = document.querySelector(`.progress[data-file="${CSS.escape(file)}"]`);
  if (!row) {
    row = document.createElement('div');
    row.className = 'progress';
    row.dataset.file = file;
    row.innerHTML = '<div class="label"><span></span><span></span></div><div class="bar"><div></div></div>';
    row.querySelector('.label span').textContent = file.split('/').pop();
    $('progress-list').appendChild(row);
  }
  return row;
}

// ---------- model loading ----------
const hasWebGPU = !!navigator.gpu && !!(await navigator.gpu.requestAdapter().catch(() => null));
const device = hasWebGPU ? 'webgpu' : 'wasm';
const dtype = hasWebGPU
  ? { encoder_model: 'fp16', decoder_model_merged: 'fp16' }
  : { encoder_model: 'q4', decoder_model_merged: 'q4' };
if (!hasWebGPU) {
  $('load-error').hidden = false;
  $('load-error').textContent = 'WebGPU is not available in this browser, so the model runs on the CPU (4-bit weights, much slower). Chrome, Edge or Safari 26 use the GPU.';
}

let processor, tokenizer, model, vad;
try {
  const onProgress = p => {
    // Only show the large model files; config and tokenizer files are tiny.
    if (!p.file || !/\.onnx(_data)?$/.test(p.file)) return;
    const row = progressRow(p.file);
    if (p.status === 'progress' && p.total) {
      row.querySelector('.bar > div').style.width = `${p.progress.toFixed(1)}%`;
      row.querySelector('.label span:last-child').textContent = `${formatMB(p.loaded)} / ${formatMB(p.total)}`;
    } else if (p.status === 'done') {
      row.querySelector('.bar > div').style.width = '100%';
      row.querySelector('.label span:last-child').textContent = 'done';
    }
  };
  [processor, tokenizer, model, vad] = await Promise.all([
    AutoProcessor.from_pretrained(MODEL_ID),
    AutoTokenizer.from_pretrained(MODEL_ID),
    WhisperForConditionalGeneration.from_pretrained(MODEL_ID, { device, dtype, progress_callback: onProgress }),
    SileroVAD.create(), // vendored Silero VAD v5.1.2 (2.3 MB, assets/)
  ]);
} catch (err) {
  console.error(err);
  $('load-title').textContent = 'The model could not be loaded';
  $('load-error').hidden = false;
  $('load-error').textContent = `${err.message || err}. Reload the page to try again.`;
  throw err;
}

const tokenId = t => tokenizer.convert_tokens_to_ids(t);
const SOT = tokenId('<|startoftranscript|>');
const TRANSCRIBE = tokenId('<|transcribe|>');
const NO_TIMESTAMPS = tokenId('<|notimestamps|>');

for (const [code, name] of LANGUAGES) {
  const opt = new Option(name, code);
  $('language').add(opt);
}
// The model has no language detection (the Python package always takes a
// language too): default to the last choice, else the browser's language.
const LANG_KEY = 'crisperwhisper.language';
let savedLang = null;
try { savedLang = localStorage.getItem(LANG_KEY); } catch {}
const browserLang = (navigator.language || '').slice(0, 2);
const known = c => LANGUAGES.some(([code]) => code === c);
$('language').value = known(savedLang) ? savedLang : known(browserLang) ? browserLang : 'en';
$('language').addEventListener('change', () => { try { localStorage.setItem(LANG_KEY, $('language').value); } catch {} });

$('loading').hidden = true;
$('input').hidden = false;

// ---------- transcription ----------
/** Prompt ids: mode tags [+ <ctx> words <ectx>], then <|sot|> <|lang|> <|transcribe|> <|notimestamps|> (prompt.py). */
function buildPrompt(langId, context) {
  let text = MODE_TAGS;
  if (context) text += ` <ctx> ${context} <ectx>`;
  return [...tokenizer.encode(text, { add_special_tokens: false }), SOT, langId, TRANSCRIBE, NO_TIMESTAMPS];
}

/** Decode one window; returns words with window-local start times (seconds). */
async function transcribeWindow(inputs, prompt, windowDuration) {
  const out = await model.generate({
    ...inputs, decoder_input_ids: prompt, return_token_timestamps: true, max_new_tokens: MAX_NEW_TOKENS,
    // Encoder frames (20 ms each) that hold real audio: DTW ignores the zero padding after them.
    num_frames: Math.ceil(windowDuration / 0.02),
  });
  const ids = Array.from(out.sequences.tolist()[0], Number).slice(prompt.length);
  const times = Array.from(out.token_timestamps.tolist()[0], Number).slice(prompt.length);
  // return_token_timestamps makes generate() return a dict and keep the KV cache
  // alive on the GPU; free it, or GPU memory grows with every window.
  await out.past_key_values?.dispose?.();
  const words = [];
  ids.forEach((id, i) => {
    const piece = tokenizer.decode([id], { skip_special_tokens: false });
    if (CONTROL_TOKEN.test(piece.trim())) return;
    const startsWord = /^\s/.test(piece) || words.length === 0;
    const text = piece.trim();
    if (!text) { if (words.length) words.at(-1).open = false; return; } // explicit space token
    if (startsWord || words.at(-1).open === false) words.push({ text, start: times[i], open: true });
    else words.at(-1).text += text;
  });
  // A word ends where the next one starts (DTW gives token start times only).
  words.forEach((w, i) => { w.end = i + 1 < words.length ? words[i + 1].start : Math.min(windowDuration, w.start + 0.5); });
  return words.map(({ text, start, end }) => ({ text, start, end }));
}

/** Overlap-aware drop (continuation.py `_overlap_drop_index`): index up to which words are confirmed. */
function keepIndex(words, isLast) {
  const n = words.length;
  if (isLast) return n;
  const legacy = Math.max(n - DROP_WORDS, 0);
  let firstOverlap = n;
  for (let i = 0; i < n; i++) if (words[i].start >= STRIDE_S) { firstOverlap = i; break; }
  return Math.max(legacy, firstOverlap);
}

/** Words -> utterances: a new row after sentence-final punctuation. */
function toSegments(words) {
  const segments = [];
  let cur = [];
  for (const w of words) {
    cur.push(w);
    if (/[.?!…]["»”)]*$/.test(w.text)) { segments.push(cur); cur = []; }
  }
  if (cur.length) segments.push(cur);
  // lastStart: start of the utterance's last word, the anchor for trimming its end to speech.
  return segments.map(ws => ({ start: ws[0].start, end: ws.at(-1).end, lastStart: ws.at(-1).start, text: ws.map(w => w.text).join(' ') }));
}

function render() {
  const tbody = $('rows');
  tbody.textContent = '';
  for (const sg of state.segments) {
    const tr = tbody.insertRow();
    tr.insertCell().textContent = formatClock(sg.start);
    tr.insertCell().textContent = formatClock(sg.end);
    tr.insertCell().textContent = sg.text;
    tr.cells[0].className = tr.cells[1].className = 't';
  }
  $('output').hidden = state.segments.length === 0;
}

async function transcribeFile(file) {
  $('drop').classList.add('busy');
  $('status').textContent = `Decoding ${file.name}…`;
  state.segments = []; state.fileName = file.name; render();
  const started = performance.now();
  try {
    const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
    const audio = (await ctx.decodeAudioData(await file.arrayBuffer())).getChannelData(0);
    ctx.close();
    // Speech regions (Silero VAD) for trimming utterance start/end times.
    $('tprogress').hidden = false;
    const vadStarted = performance.now();
    const speech = await detectSpeech(vad, audio, f => {
      $('tprogress-bar').style.width = `${(100 * f).toFixed(1)}%`;
      $('tprogress-text').textContent = `Detecting speech: ${Math.round(100 * f)}%`;
      $('tprogress-eta').textContent = '';
    });
    const vadSeconds = (performance.now() - vadStarted) / 1000;
    console.info(`[vad] ${speech.length} speech regions in ${vadSeconds.toFixed(1)} s`);
    const segmentsFrom = words => trimToSpeech(toSegments(words), speech);
    const total = audio.length / SAMPLE_RATE;
    const nWindows = total <= CHUNK_S ? 1 : Math.ceil((total - CHUNK_S) / STRIDE_S) + 1;

    const langId = tokenId(`<|${$('language').value}|>`);
    const confirmed = [];
    for (let i = 0; i < nWindows; i++) {
      const start = i * STRIDE_S;
      const window = audio.subarray(Math.round(start * SAMPLE_RATE), Math.round(Math.min(total, start + CHUNK_S) * SAMPLE_RATE));
      $('status').textContent = `Transcribing ${file.name}…`;
      showProgress(i, nWindows, started);
      const inputs = await processor(window);
      const context = i === 0 ? null : confirmed.slice(-CONTEXT_WORDS).map(w => w.text).join(' ') || null;
      const words = await transcribeWindow(inputs, buildPrompt(langId, context), window.length / SAMPLE_RATE);
      const kept = words.slice(0, keepIndex(words, i === nWindows - 1));
      for (const w of kept) confirmed.push({ text: w.text, start: w.start + start, end: Math.min(w.end + start, total) });
      state.segments = segmentsFrom(confirmed);
      render();
      showProgress(i + 1, nWindows, started);
    }
    // Keep the timeline monotonic at window seams.
    for (let i = 1; i < confirmed.length; i++) confirmed[i].start = Math.max(confirmed[i].start, confirmed[i - 1].start);
    state.segments = segmentsFrom(confirmed);
    render();
    $('status').textContent = `Done: ${file.name}, ${formatClock(total)} of audio in ${((performance.now() - started) / 1000).toFixed(1)} s.`;
  } catch (err) {
    console.error(err);
    $('status').textContent = `Transcription failed: ${err.message || err}`;
  } finally {
    $('drop').classList.remove('busy');
    $('tprogress').hidden = true;
    $('file').value = '';
  }
}

// ---------- UI wiring ----------
$('file').addEventListener('change', e => { const f = e.target.files?.[0]; if (f) transcribeFile(f); });
for (const ev of ['dragenter', 'dragover']) $('drop').addEventListener(ev, e => { e.preventDefault(); $('drop').classList.add('over'); });
for (const ev of ['dragleave', 'drop']) $('drop').addEventListener(ev, () => $('drop').classList.remove('over'));
$('drop').addEventListener('drop', e => {
  e.preventDefault();
  const f = e.dataTransfer?.files?.[0];
  if (f) transcribeFile(f);
});
$('eaf').addEventListener('click', () => {
  const xml = buildEaf([{ name: fileStem(state.fileName), mediaFileName: state.fileName, segments: state.segments }]);
  downloadText(xml, `${fileStem(state.fileName)}.eaf`);
});
$('copy').addEventListener('click', () => {
  const tsv = ['start\tend\ttext', ...state.segments.map(s => `${s.start.toFixed(2)}\t${s.end.toFixed(2)}\t${s.text}`)].join('\n');
  navigator.clipboard.writeText(tsv);
});

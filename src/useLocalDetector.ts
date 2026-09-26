import { useEffect, useRef, useState } from 'react';
import { DEFAULT_SCORE_THRESHOLD, pickModel } from './detectorConfig';
import { LocalTracker } from './localTracker';
import type { TrackedBox } from './types';
import type { WorkerRequest, WorkerResponse } from './detector.worker';

const DEFAULT_INTERVAL_MS = 333; // 3 fps default
// iOS WebKit kills tabs that sustain ~30%+ CPU for several seconds, so back
// off the inference cadence on iPhone/iPad where WASM is the only path.
const MOBILE_WEBKIT_INTERVAL_MS = 666; // ~1.5 fps

function isIOSWebKit(): boolean {
  if (typeof navigator === 'undefined') return false;
  const ua = navigator.userAgent;
  return (
    /iPad|iPhone|iPod/.test(ua) ||
    (ua.includes('Mac') &&
      typeof document !== 'undefined' &&
      'ontouchend' in document)
  );
}

function readIntervalOverride(): number {
  const base = isIOSWebKit() ? MOBILE_WEBKIT_INTERVAL_MS : DEFAULT_INTERVAL_MS;
  if (typeof window === 'undefined') return base;
  const fpsParam = new URLSearchParams(window.location.search).get('fps');
  const fps = fpsParam ? Number(fpsParam) : NaN;
  if (Number.isFinite(fps) && fps > 0) return Math.round(1000 / fps);
  return base;
}

let nextSeq = 0;

function readParam(name: string): string | null {
  if (typeof window === 'undefined') return null;
  return new URLSearchParams(window.location.search).get(name);
}

// `?conf=N` overrides the score needed to start a new box.
function readScoreThreshold(): number {
  const n = Number(readParam('conf') ?? NaN);
  return Number.isFinite(n) && n > 0 && n < 1 ? n : DEFAULT_SCORE_THRESHOLD;
}

export function useLocalDetector(opts: {
  videoEl: HTMLVideoElement | null;
  enabled: boolean;
}) {
  // boxes is exposed as a ref (not state) so that per-frame track
  // extrapolation doesn't re-render the entire App tree — the overlay rAF loop
  // reads the ref directly.
  const boxesRef = useRef<TrackedBox[]>([]);
  const [ready, setReady] = useState(false);
  const [backend, setBackend] = useState<'wasm' | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState({
    inferences: 0,
    lastError: '',
    maxScore: 0,
    rawCount: 0,
    keptCount: 0,
    inferMs: 0,
    fps: 0,
    model: '',
  });
  const workerRef = useRef<Worker | null>(null);
  const trackerRef = useRef<LocalTracker | null>(null);
  if (!trackerRef.current) trackerRef.current = new LocalTracker();

  // Spin up the worker and load the model lazily — only when first enabled,
  // so users who never start never download the ONNX file.
  useEffect(() => {
    if (!opts.enabled || workerRef.current || error) return;
    const worker = new Worker(new URL('./detector.worker.ts', import.meta.url), {
      type: 'module',
    });
    workerRef.current = worker;
    const onMessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data;
      if (msg.type === 'loaded') {
        setBackend(msg.backend);
        setReady(true);
      } else if (msg.type === 'error' && msg.phase === 'load') {
        setError(msg.message || 'モデル読み込み失敗');
      }
    };
    worker.addEventListener('message', onMessage);
    worker.addEventListener('error', (e) => {
      setError(e.message || 'Worker 起動失敗');
    });
    const req: WorkerRequest = {
      type: 'load',
      modelUrl: pickModel(readParam('model')).url,
      scoreThreshold: readScoreThreshold(),
    };
    worker.postMessage(req);
    // The worker (and its loaded session) is intentionally kept for the page's
    // lifetime, same as the old module-level session.
  }, [opts.enabled, error]);

  useEffect(() => {
    const worker = workerRef.current;
    if (!opts.enabled || !ready || !opts.videoEl || !worker) return;
    const video = opts.videoEl;
    const tracker = trackerRef.current!;
    const scoreThreshold = readScoreThreshold();
    const intervalMs = readIntervalOverride();
    const model = pickModel(readParam('model')).name;
    let stopped = false;
    // Seq of the request we're waiting on (-1 = none). A result from an
    // earlier effect run (e.g. after a video swap) carries an old seq and is
    // dropped instead of polluting the fresh tracker.
    let pendingSeq = -1;
    let lastRun = 0;
    let inferences = 0;
    let inferMsEma = 0;
    let lastError = '';
    let last = { maxScore: 0, rawCount: 0, keptCount: 0 };
    let windowStart = performance.now();
    let windowCount = 0;
    let fps = 0;

    const onMessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data;
      if (msg.type === 'result') {
        if (msg.seq !== pendingSeq) return;
        pendingSeq = -1;
        if (stopped) return;
        tracker.update(msg.boxes, msg.capturedAt, scoreThreshold);
        inferences++;
        windowCount++;
        inferMsEma = inferMsEma ? inferMsEma * 0.8 + msg.inferMs * 0.2 : msg.inferMs;
        last = msg;
      } else if (msg.type === 'error' && msg.phase === 'detect') {
        if (msg.seq !== pendingSeq) return;
        pendingSeq = -1;
        lastError = msg.message.slice(0, 120);
        console.error('Local detect error', msg.message);
      }
    };
    worker.addEventListener('message', onMessage);

    const statsTimer = setInterval(() => {
      if (stopped) return;
      const now = performance.now();
      fps = (windowCount * 1000) / Math.max(1, now - windowStart);
      windowStart = now;
      windowCount = 0;
      setStats({
        inferences,
        lastError,
        maxScore: last.maxScore,
        rawCount: last.rawCount,
        keptCount: last.keptCount,
        inferMs: inferMsEma,
        fps,
        model,
      });
    }, 1000);

    const loop = async (ts: number) => {
      if (stopped) return;
      // Pause inference while the tab is hidden — sustained WASM work in the
      // background is what triggers iOS WebKit's memory-pressure tab kill.
      if (typeof document !== 'undefined' && document.hidden) {
        requestAnimationFrame(loop);
        return;
      }
      boxesRef.current = tracker.snapshot(performance.now());
      if (pendingSeq < 0 && video.videoWidth && ts - lastRun >= intervalMs) {
        const seq = nextSeq++;
        pendingSeq = seq;
        lastRun = ts;
        const capturedAt = performance.now();
        try {
          const frame = await createImageBitmap(video);
          if (stopped) {
            frame.close();
          } else {
            const req: WorkerRequest = {
              type: 'detect',
              seq,
              frame,
              capturedAt,
            };
            worker.postMessage(req, [frame]);
          }
        } catch (e) {
          pendingSeq = -1;
          lastError = (e instanceof Error ? e.message : String(e)).slice(0, 120);
        }
      }
      if (!stopped) requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);

    return () => {
      stopped = true;
      worker.removeEventListener('message', onMessage);
      clearInterval(statsTimer);
      boxesRef.current = [];
      tracker.reset();
    };
  }, [opts.enabled, ready, opts.videoEl]);

  return { boxesRef, ready, backend, error, stats };
}

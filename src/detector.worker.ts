/// <reference lib="webworker" />
import { detect, lastStats, loadModel } from './yolo11';
import type { LiveBox } from './types';

// Runs DEIMv2 off the main thread. On iOS a single WASM inference takes
// ~150–300ms; on the main thread that froze the overlay's rAF loop for the
// whole run, so boxes could only ever jump between inference results. Here the
// main thread keeps drawing (and extrapolating tracks) while this worker
// grinds.

export type WorkerRequest =
  | { type: 'load'; modelUrl: string; scoreThreshold: number }
  | { type: 'detect'; seq: number; frame: ImageBitmap; capturedAt: number };

export type WorkerResponse =
  | { type: 'loaded'; backend: 'wasm' }
  | {
      type: 'result';
      seq: number;
      boxes: LiveBox[];
      capturedAt: number;
      inferMs: number;
      maxScore: number;
      rawCount: number;
      keptCount: number;
    }
  | { type: 'error'; phase: 'load'; message: string }
  | { type: 'error'; phase: 'detect'; seq: number; message: string };

const ctx = self as unknown as DedicatedWorkerGlobalScope;

// Handle messages strictly one at a time: onnxruntime sessions don't support
// concurrent run() calls, and a stale request can still be queued when the
// main thread restarts its loop.
let queue: Promise<void> = Promise.resolve();
ctx.onmessage = (e: MessageEvent<WorkerRequest>) => {
  queue = queue.then(() => handle(e.data));
};

async function handle(msg: WorkerRequest) {
  if (msg.type === 'load') {
    try {
      const r = await loadModel({
        modelUrl: msg.modelUrl,
        scoreThreshold: msg.scoreThreshold,
      });
      post({ type: 'loaded', backend: r.backend });
    } catch (err) {
      post({ type: 'error', phase: 'load', message: errMessage(err) });
    }
    return;
  }

  const t0 = performance.now();
  try {
    const boxes = await detect(msg.frame);
    post({
      type: 'result',
      seq: msg.seq,
      boxes,
      capturedAt: msg.capturedAt,
      inferMs: performance.now() - t0,
      ...lastStats,
    });
  } catch (err) {
    post({
      type: 'error',
      phase: 'detect',
      seq: msg.seq,
      message: errMessage(err),
    });
  } finally {
    msg.frame.close();
  }
}

function post(r: WorkerResponse) {
  ctx.postMessage(r);
}

function errMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

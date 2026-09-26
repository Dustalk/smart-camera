import * as ort from 'onnxruntime-web/wasm';
import type { LiveBox } from './types';
import { LOW_SCORE_FLOOR } from './detectorConfig';

// DEIMv2-N: a DETR-family detector with the HGNetv2 backbone, trained on
// COCO. 3.6M params, ~4.4MB INT8, Apache 2.0 licensed. End-to-end ONNX —
// the postprocessor is baked into the graph so we get labels/boxes/scores
// directly, no per-anchor decoding or NMS on the JS side.
//
// Upgraded from DEIMv2-Pico (1.5M, AP 38.5) for better localization. N is
// 43.0 AP — the +4.5 AP closes most of the gap to S (50.9 AP) at less than
// half the parameter count. Classification head is still treated as
// unreliable (labels are clobbered to "物体" / hidden); the upgrade is
// purely for cleaner bboxes.
//
// File name `yolo11.ts` is kept for git/history continuity. Preprocess is
// still YOLO-style letterbox (114-gray pad + RGB CHW + /255) which DETR
// also accepts.
const INPUT_SIZE = 640;
// DEIMv2-Pico is COCO-pretrained so labels[i] is a COCO class id in [0, 80).
// Empirically at INT8 + 1.5M params the classification head is noisy (e.g.
// posters get tagged as "toothbrush"). The bbox itself is still well placed,
// and Gemini does the real identification in /api/refine-items, so we
// throw the COCO label away and show a generic "物体" on every box. This
// mirrors what we did with FastSAM and keeps the user from seeing
// confidently-wrong labels mid-session.
const GENERIC_LABEL = '物体';
// DEIMv2's decoder returns a fixed number of object queries (200 per the
// Pico config). The deploy-mode postprocessor returns the top scoring 300
// across the batch — already sorted by score descending. We just iterate
// until scores drop below threshold.
const MAX_QUERIES = 300;
let session: ort.InferenceSession | null = null;
let activeBackend: 'wasm' | null = null;

const inputBuffer = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
// orig_target_sizes is a fixed [[640, 640]] for our use — we letterbox the
// video frame into 640² and ask the postprocessor to scale boxes back to
// 640² space, then we un-letterbox to video pixels ourselves.
const origSizes = new BigInt64Array([BigInt(INPUT_SIZE), BigInt(INPUT_SIZE)]);

ort.env.wasm.wasmPaths =
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
ort.env.wasm.numThreads = 1;

let scoreFloor = LOW_SCORE_FLOOR;

export async function loadModel(opts: {
  modelUrl: string;
  scoreThreshold: number;
}): Promise<{ backend: 'wasm' }> {
  if (session && activeBackend) return { backend: activeBackend };
  scoreFloor = Math.min(LOW_SCORE_FLOOR, opts.scoreThreshold);
  const modelUrl = opts.modelUrl;
  session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  activeBackend = 'wasm';
  return { backend: activeBackend };
}

type LetterboxMeta = {
  vw: number;
  vh: number;
  scale: number;
  padX: number;
  padY: number;
};

// OffscreenCanvas so preprocessing works inside the detector worker.
let scratchCtx: OffscreenCanvasRenderingContext2D | null = null;

function preprocess(frame: ImageBitmap): LetterboxMeta {
  const vw = frame.width;
  const vh = frame.height;

  if (!scratchCtx) {
    scratchCtx = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE).getContext('2d', {
      willReadFrequently: true,
    })!;
  }
  const ctx = scratchCtx;

  const scale = Math.min(INPUT_SIZE / vw, INPUT_SIZE / vh);
  const dw = vw * scale;
  const dh = vh * scale;
  const padX = (INPUT_SIZE - dw) / 2;
  const padY = (INPUT_SIZE - dh) / 2;

  ctx.fillStyle = 'rgb(114, 114, 114)';
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  ctx.drawImage(frame, padX, padY, dw, dh);

  const data = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const stride = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    inputBuffer[j] = data[i] / 255;
    inputBuffer[j + stride] = data[i + 1] / 255;
    inputBuffer[j + 2 * stride] = data[i + 2] / 255;
  }

  return { vw, vh, scale, padX, padY };
}

// Diagnostic counters from the last postprocess pass — exposed to the
// debug overlay.
export const lastStats = { maxScore: 0, rawCount: 0, keptCount: 0 };

// DEIMv2 deploy-mode outputs are already postprocessed: scores sorted
// descending across the batch's queries, boxes in input-image pixel space
// (xyxy), labels as integer class IDs. We just threshold and un-letterbox.
function postprocess(
  labels: BigInt64Array | Int32Array,
  boxes: Float32Array,
  scores: Float32Array,
  meta: LetterboxMeta,
): LiveBox[] {
  const out: LiveBox[] = [];
  const n = Math.min(scores.length, MAX_QUERIES);
  lastStats.maxScore = n > 0 ? scores[0] : 0;
  let raw = 0;

  for (let i = 0; i < n; i++) {
    const score = scores[i];
    // Scores are descending — bail as soon as we drop below the floor.
    if (score < scoreFloor) break;
    raw++;

    const lx1 = boxes[i * 4 + 0];
    const ly1 = boxes[i * 4 + 1];
    const lx2 = boxes[i * 4 + 2];
    const ly2 = boxes[i * 4 + 3];

    // Un-letterbox: boxes come back in the 640² input frame; map back to
    // the original video pixel space.
    const x1 = Math.max(0, Math.min(meta.vw, (lx1 - meta.padX) / meta.scale));
    const y1 = Math.max(0, Math.min(meta.vh, (ly1 - meta.padY) / meta.scale));
    const x2 = Math.max(0, Math.min(meta.vw, (lx2 - meta.padX) / meta.scale));
    const y2 = Math.max(0, Math.min(meta.vh, (ly2 - meta.padY) / meta.scale));
    if (x2 <= x1 || y2 <= y1) continue;

    const classId = Number(labels[i]);
    out.push({
      bbox: [x1, y1, x2, y2],
      score,
      classId,
      label: GENERIC_LABEL,
    });
  }

  lastStats.rawCount = raw;
  lastStats.keptCount = out.length;
  return out;
}

export async function detect(frame: ImageBitmap): Promise<LiveBox[]> {
  if (!session) throw new Error('Model not loaded');
  if (!frame.width) return [];

  const meta = preprocess(frame);
  const images = new ort.Tensor('float32', inputBuffer, [
    1,
    3,
    INPUT_SIZE,
    INPUT_SIZE,
  ]);
  const origSizesTensor = new ort.Tensor('int64', origSizes, [1, 2]);

  let results: ort.InferenceSession.OnnxValueMapType | null = null;
  try {
    results = await session.run({
      images,
      orig_target_sizes: origSizesTensor,
    });
    const labels = results['labels'].data as BigInt64Array | Int32Array;
    const boxes = results['boxes'].data as Float32Array;
    const scores = results['scores'].data as Float32Array;
    return postprocess(labels, boxes, scores, meta);
  } finally {
    images.dispose();
    origSizesTensor.dispose();
    if (results) {
      for (const name of session.outputNames) {
        const t = results[name];
        if (t && typeof t.dispose === 'function') t.dispose();
      }
    }
  }
}

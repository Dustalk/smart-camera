// Detector settings shared by the main thread and the detector worker. Kept
// out of yolo11.ts so the main bundle doesn't pull in onnxruntime-web.

// Bias for recall: false positives are silently ignored (user just doesn't
// tap), but missed objects can't be selected. `?conf=N` overrides.
export const DEFAULT_SCORE_THRESHOLD = 0.25;
// ByteTrack-style second stage: detections between this floor and the main
// threshold are returned too, but LocalTracker only uses them to keep an
// existing track alive — they never create a new box on their own.
export const LOW_SCORE_FLOOR = 0.12;

// Trial #6 (iOS ceiling probe): `?model=s` loads DEIMv2-S (DINOv3 ViT-Tiny,
// 9.7M params, COCO AP 50.9, ~11.8MB INT8) instead of the default N (3.6M,
// AP 43.0, 4.4MB). Lets us A/B both on the same device without a redeploy —
// we're measuring whether the ~12MB S model loads and runs without iOS
// Safari memory-pressure kills, and whether its bboxes are visibly better.
// Same I/O (images + orig_target_sizes → labels/boxes/scores) and same
// /255 letterbox preprocess, so it's a pure modelUrl swap.
export const MODELS = {
  n: { name: 'deimv2-n', url: '/models/deimv2_n_640_uint8.onnx' },
  s: { name: 'deimv2-s', url: '/models/deimv2_s_640_uint8.onnx' },
} as const;

export function pickModel(variant: string | null) {
  return variant === 's' ? MODELS.s : MODELS.n;
}

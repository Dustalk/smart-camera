import type { LiveBox, TrackedBox } from './types';

// ByteTrack-flavoured tracker for ~1.5–3 fps inference.
//
// - Each track carries a constant-velocity alpha-beta filter (a fixed-gain
//   Kalman) on its box centre. `snapshot(now)` extrapolates every track to the
//   current time, so the overlay moves smoothly between inference results and
//   the ~150–300ms inference latency is compensated (detections are stamped
//   with their capture time, not their completion time).
// - Association is two-stage: confident detections first, then low-score
//   detections may only extend an existing track (never start one). This keeps
//   a box alive through the frames where the detector's confidence dips.
// - Lost tracks are remembered for TRACK_GC_MS and matched against their
//   predicted position, so an object that disappears briefly comes back with
//   the same instance_id instead of becoming a second cart entry.

// Forget a track this long after its last match.
const TRACK_GC_MS = 3000;
// Keep drawing an unmatched track for this long (bridges 1–2 missed frames).
const COAST_SHOW_MS = 900;
// Don't extrapolate further than this past the last match — a coasting track
// freezes instead of sliding off the screen.
const MAX_PREDICT_MS = 500;
// Stage 1 (confident detections) and stage 2 (low-score detections) IoU gates
// against the predicted box. Stage 2 is stricter because those detections are
// more likely to be noise.
const MATCH_IOU_HIGH = 0.3;
const MATCH_IOU_LOW = 0.5;
// Filter gains. ALPHA close to 1 trusts the detector (low lag); BETA sets how
// fast velocity reacts. Size is smoothed without a velocity term.
const ALPHA = 0.75;
const BETA = 0.3;
const SIZE_ALPHA = 0.6;
// Clamp velocity (px/ms) so one bad match can't fling a box across the frame.
const MAX_SPEED = 2;

type Bbox = [number, number, number, number];

type TrackerState = {
  label: string;
  classId: number;
  score: number;
  // Centre, size and centre velocity in video px / ms, valid at `updatedAt`.
  cx: number;
  cy: number;
  w: number;
  h: number;
  vx: number;
  vy: number;
  updatedAt: number;
};

function iou(a: Bbox, b: Bbox): number {
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]);
  const iy2 = Math.min(a[3], b[3]);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  const aArea = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const bArea = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = aArea + bArea - inter;
  return union <= 0 ? 0 : inter / union;
}

function clampSpeed(v: number): number {
  return Math.max(-MAX_SPEED, Math.min(MAX_SPEED, v));
}

function predictBbox(t: TrackerState, atMs: number): Bbox {
  const dt = Math.max(0, Math.min(MAX_PREDICT_MS, atMs - t.updatedAt));
  const cx = t.cx + t.vx * dt;
  const cy = t.cy + t.vy * dt;
  return [cx - t.w / 2, cy - t.h / 2, cx + t.w / 2, cy + t.h / 2];
}

export class LocalTracker {
  private trackers = new Map<number, TrackerState>();
  private nextId = 1;

  // `capturedAt` is when the analysed frame was grabbed (performance.now()),
  // not when inference finished. Detections scoring below `highThreshold` are
  // only used to extend existing tracks.
  update(rawBoxes: LiveBox[], capturedAt: number, highThreshold: number) {
    for (const [id, t] of this.trackers) {
      if (capturedAt - t.updatedAt > TRACK_GC_MS) this.trackers.delete(id);
    }

    const predicted = new Map<number, Bbox>();
    for (const [id, t] of this.trackers) {
      predicted.set(id, predictBbox(t, capturedAt));
    }

    const high: number[] = [];
    const low: number[] = [];
    rawBoxes.forEach((d, i) =>
      (d.score >= highThreshold ? high : low).push(i),
    );

    const usedTracker = new Set<number>();
    const detToTracker = new Map<number, number>();
    const associate = (detIdxs: number[], minIou: number) => {
      type Pair = { iou: number; trackerId: number; detIdx: number };
      const pairs: Pair[] = [];
      for (const i of detIdxs) {
        const det = rawBoxes[i];
        for (const [id, t] of this.trackers) {
          if (usedTracker.has(id) || t.label !== det.label) continue;
          const v = iou(predicted.get(id)!, det.bbox);
          if (v >= minIou) pairs.push({ iou: v, trackerId: id, detIdx: i });
        }
      }
      pairs.sort((a, b) => b.iou - a.iou);
      for (const p of pairs) {
        if (usedTracker.has(p.trackerId) || detToTracker.has(p.detIdx)) {
          continue;
        }
        usedTracker.add(p.trackerId);
        detToTracker.set(p.detIdx, p.trackerId);
      }
    };
    associate(high, MATCH_IOU_HIGH);
    associate(low, MATCH_IOU_LOW);

    for (const [detIdx, id] of detToTracker) {
      this.correct(this.trackers.get(id)!, rawBoxes[detIdx], capturedAt);
    }
    for (const i of high) {
      if (detToTracker.has(i)) continue;
      const det = rawBoxes[i];
      const [x1, y1, x2, y2] = det.bbox;
      this.trackers.set(this.nextId++, {
        label: det.label,
        classId: det.classId,
        score: det.score,
        cx: (x1 + x2) / 2,
        cy: (y1 + y2) / 2,
        w: x2 - x1,
        h: y2 - y1,
        vx: 0,
        vy: 0,
        updatedAt: capturedAt,
      });
    }
  }

  // Tracks to draw right now, extrapolated to `nowMs`.
  snapshot(nowMs: number): TrackedBox[] {
    const out: TrackedBox[] = [];
    for (const [id, t] of this.trackers) {
      if (nowMs - t.updatedAt > COAST_SHOW_MS) continue;
      out.push({
        bbox: predictBbox(t, nowMs),
        label: t.label,
        classId: t.classId,
        score: t.score,
        instance_id: id,
      });
    }
    return out;
  }

  reset() {
    this.trackers.clear();
    this.nextId = 1;
  }

  private correct(t: TrackerState, det: LiveBox, atMs: number) {
    const dt = Math.max(1, atMs - t.updatedAt);
    const [px1, py1, px2, py2] = predictBbox(t, atMs);
    const pcx = (px1 + px2) / 2;
    const pcy = (py1 + py2) / 2;
    const [x1, y1, x2, y2] = det.bbox;
    const rx = (x1 + x2) / 2 - pcx;
    const ry = (y1 + y2) / 2 - pcy;
    t.cx = pcx + ALPHA * rx;
    t.cy = pcy + ALPHA * ry;
    t.vx = clampSpeed(t.vx + (BETA * rx) / dt);
    t.vy = clampSpeed(t.vy + (BETA * ry) / dt);
    t.w += SIZE_ALPHA * (x2 - x1 - t.w);
    t.h += SIZE_ALPHA * (y2 - y1 - t.h);
    t.score = det.score;
    t.classId = det.classId;
    t.updatedAt = atMs;
  }
}

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useCamera } from './useCamera';
import { useLocalDetector } from './useLocalDetector';
import {
  captureBoxCropJpeg,
  captureFrameJpeg,
  cropJpegBase64,
} from './captureSnapshot';
import { Button } from './ui/Button';
import { DetectOverlay } from './live/DetectOverlay';
import { CartView } from './cart/CartView';
import { useLiveSession } from './live/useLiveSession';
import type { RetainedFrame } from './live/useLiveSession';
import { createCartHandlers } from './live/cartTools';
import type { CartEntry, RefinedItem, TrackedBox } from './types';
import './App.css';

type Phase = 'idle' | 'live' | 'refining' | 'cart';

const MAX_SELECTIONS = 30;

const DEBUG =
  typeof window !== 'undefined' &&
  new URLSearchParams(window.location.search).get('debug') === '1';

export default function App() {
  const [phase, setPhase] = useState<Phase>('idle');
  // Cart keyed by instance_id from LocalTracker — so tapping the same physical
  // object twice is deduped, while distinct instances of the same label
  // aggregate as a count after refining.
  const [cart, setCart] = useState<Map<number, CartEntry>>(new Map());
  const [refineError, setRefineError] = useState<string | null>(null);
  const [tooManyVisible, setTooManyVisible] = useState(false);
  const tooManyTimerRef = useRef<number | null>(null);

  const camera = useCamera();
  const localDetector = useLocalDetector({
    videoEl: camera.videoEl,
    enabled: phase === 'live',
  });

  const {
    boxesRef: localBoxesRef,
    ready: detectorReady,
    backend: detectorBackend,
    error: detectorError,
    stats: detectorStats,
  } = localDetector;
  const cameraError = camera.error;

  const cartRef = useRef(cart);
  useEffect(() => {
    cartRef.current = cart;
  });

  // Set of selected instance_ids for the overlay's highlight, recomputed only
  // when the cart changes.
  const selectedIds = useMemo(() => new Set(cart.keys()), [cart]);
  const cartCount = cart.size;

  // Voice-mode cart talk. Voice entries get unique *negative* instance_ids so
  // they never collide with the tracker's positive ids; tools read the live
  // cart via cartRef and mutate it through setCart.
  const voiceIdRef = useRef(-1);
  // Last frame sent to the Live session. Owned here (not by useLiveSession)
  // because the cart handlers below are created before the hook runs — the
  // hook writes it, the add_to_cart handler reads it to crop the object the
  // model referenced.
  const lastFrameRef = useRef<RetainedFrame | null>(null);
  // Rebuilt each render (cheap); useLiveSession mirrors it into a ref, so a
  // fresh object never reconnects the session. The handlers only read cartRef/
  // voiceIdRef at call time (in tool callbacks), never during render.
  const cartHandlers = createCartHandlers({
    getCart: () => cartRef.current,
    setCart,
    nextVoiceId: () => voiceIdRef.current--,
    max: MAX_SELECTIONS,
    getLastFrame: () => lastFrameRef.current,
    getVideoEl: () => camera.videoEl,
  });
  const talk = useLiveSession({
    videoEl: camera.videoEl,
    handlers: cartHandlers,
    lastFrameRef,
  });
  const talkActive = talk.status === 'active' || talk.status === 'connecting';
  const toggleTalk = useCallback(() => {
    if (talkActive) talk.stop();
    else void talk.start();
  }, [talkActive, talk]);

  const handleStart = useCallback(async () => {
    setRefineError(null);
    setPhase('live');
    if (!camera.active) {
      await camera.start();
    }
  }, [camera]);

  // A tap landed inside a detection box: add it to the cart (deduped, capped),
  // cropping the box (with margin) out of the current frame as the snapshot.
  const handlePick = useCallback(
    (hit: TrackedBox) => {
      const video = camera.videoEl;
      if (!video) return;
      if (cartRef.current.has(hit.instance_id)) return;
      if (cartRef.current.size >= MAX_SELECTIONS) {
        setTooManyVisible(true);
        if (tooManyTimerRef.current !== null) {
          window.clearTimeout(tooManyTimerRef.current);
        }
        tooManyTimerRef.current = window.setTimeout(() => {
          setTooManyVisible(false);
          tooManyTimerRef.current = null;
        }, 2200);
        return;
      }

      let snapshot = '';
      let bbox: [number, number, number, number] = [0, 0, 1, 1];
      try {
        const crop = captureBoxCropJpeg(video, hit.bbox);
        snapshot = crop.data;
        bbox = crop.innerBox;
      } catch (err) {
        console.warn('box crop failed, falling back to full frame', err);
        try {
          snapshot = captureFrameJpeg(video);
          const vw = video.videoWidth;
          const vh = video.videoHeight;
          const [bx1, by1, bx2, by2] = hit.bbox;
          if (vw && vh) bbox = [bx1 / vw, by1 / vh, bx2 / vw, by2 / vh];
        } catch (err2) {
          console.warn('snapshot failed', err2);
        }
      }

      setCart((prev) => {
        if (prev.has(hit.instance_id)) return prev;
        const next = new Map(prev);
        next.set(hit.instance_id, {
          instance_id: hit.instance_id,
          yolo_label: hit.label,
          snapshot_b64: snapshot,
          snapshot_bbox: bbox,
          source: 'tap',
        });
        return next;
      });
    },
    [camera.videoEl],
  );

  const stopTalk = talk.stop;
  const handleStopLive = useCallback(async () => {
    stopTalk();
    camera.stop();
    const entries = Array.from(cartRef.current.values()).filter(
      (e) => (e.snapshot_b64?.length ?? 0) > 0,
    );
    if (entries.length === 0) {
      setPhase('cart');
      return;
    }
    setPhase('refining');
    setRefineError(null);
    try {
      const res = await fetch('/api/refine-items', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          items: entries.map((e) => ({
            id: e.instance_id,
            yolo_label: e.yolo_label,
            image_b64: e.snapshot_b64,
            image_mime: 'image/jpeg',
            bbox: e.snapshot_bbox,
            source: e.source,
          })),
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        if (res.status === 503) {
          throw new Error('サーバーが一時的に混み合っています。少し時間をおいて再試行してください。');
        }
        throw new Error(body?.error || `HTTP ${res.status}`);
      }
      const data = (await res.json()) as {
        items?: Array<{
          id: number;
          refined: RefinedItem;
          box_2d?: [number, number, number, number];
        }>;
      };
      // Voice items were sent as full frames; the server returns box_2d for the
      // named object, so crop the thumbnail here (bbox responsibility moved from
      // Live to this "解析" step). Do all async crops up front, then setCart once.
      const crops = new Map<
        number,
        { data: string; innerBox: [number, number, number, number] }
      >();
      await Promise.all(
        (data.items ?? []).map(async (r) => {
          if (!r.box_2d) return;
          const cur = cartRef.current.get(r.id);
          if (!cur?.snapshot_b64) return;
          try {
            const c = await cropJpegBase64(cur.snapshot_b64, r.box_2d);
            if (c) crops.set(r.id, c);
          } catch {
            // decode/crop failed — keep the full frame as the thumbnail
          }
        }),
      );
      // Snapshots are kept (not cleared) — the cart screen shows them as
      // thumbnails for later confirmation. They're margin-cropped JPEGs, so
      // 30 of them is a few MB of base64 at most.
      setCart((prev) => {
        const next = new Map(prev);
        for (const r of data.items ?? []) {
          const cur = next.get(r.id);
          if (!cur) continue;
          const crop = crops.get(r.id);
          next.set(r.id, {
            ...cur,
            refined: r.refined,
            ...(crop
              ? { snapshot_b64: crop.data, snapshot_bbox: crop.innerBox }
              : {}),
          });
        }
        return next;
      });
    } catch (err) {
      // The raw message may name the upstream model/provider, so the cart
      // banner shows a generic line; keep the detail here for debugging.
      console.warn('refine-items failed', err);
      const msg = err instanceof Error ? err.message : String(err);
      setRefineError(msg);
    } finally {
      setPhase('cart');
    }
  }, [camera, stopTalk]);

  const handleReset = useCallback(() => {
    stopTalk();
    setCart(new Map());
    setRefineError(null);
    setPhase('idle');
  }, [stopTalk]);

  const handleRemove = useCallback((ids: number[]) => {
    setCart((prev) => {
      const next = new Map(prev);
      for (const id of ids) next.delete(id);
      return next;
    });
  }, []);

  return (
    <div className="app">
      {phase === 'idle' && (
        <div className="screen idle">
          <img src="/dustalk-logo.png" alt="Dustalk" className="dustalk-logo" />
          <h1>動画アシスタント</h1>
          <p className="lead">
            カメラを起動 → 写った物体の枠をタップ、または 🎙
            で話しかけると、捨てる物がカゴに入ります。停止すると AI
            が詳細を返します。
          </p>
          {detectorError && <div className="status err">{detectorError}</div>}
          {cameraError && <div className="status err">{cameraError}</div>}
          <Button onClick={handleStart}>カメラ開始</Button>
        </div>
      )}

      {phase === 'live' && (
        <div className="screen running">
          <video
            ref={camera.videoRef}
            autoPlay
            playsInline
            muted
            className="video"
          />
          <DetectOverlay
            video={camera.videoEl}
            boxesRef={localBoxesRef}
            selectedIds={selectedIds}
            onPick={handlePick}
          />
          <div className="badge">🛒 {cartCount}</div>
          <div className="live-cart-panel">
            {cartCount === 0 ? (
              <div className="live-cart-empty">
                枠をタップ、または 🎙 で話しかけてカゴに追加
              </div>
            ) : (
              <div className="live-cart-chip">
                <span className="live-cart-chip-label">カゴ</span>
                <span className="live-cart-chip-count">×{cartCount}</span>
                <button
                  className="live-cart-chip-remove"
                  onClick={() => setCart(new Map())}
                  aria-label="すべて削除"
                >
                  ×
                </button>
              </div>
            )}
          </div>
          {tooManyVisible && (
            <div className="error">
              選択は {MAX_SELECTIONS} 個までです。停止して詳細取得に進んでください。
            </div>
          )}
          {talk.status !== 'idle' && (
            <div className={`talk-status ${talk.status}`}>
              {talk.status === 'connecting' && '接続中…'}
              {talk.status === 'active' &&
                '🎙 話しかけてください（例:「ペットボトル2つ追加して」）'}
              {talk.status === 'error' &&
                `音声エラー: ${talk.error ?? '不明'}`}
            </div>
          )}
          <button className="stop" onClick={handleStopLive}>
            停止
          </button>
          <button
            className={`talk-btn ${talkActive ? 'active' : ''}`}
            onClick={toggleTalk}
            aria-label={talkActive ? '音声モードを終了' : '音声モードを開始'}
          >
            {talkActive ? '🔴' : '🎙'}
          </button>
          {!detectorReady && !detectorError && (
            <div className="preview-tip">準備中…</div>
          )}
          {detectorError && <div className="error">エラー: {detectorError}</div>}
          {cameraError && <div className="error">{cameraError}</div>}
          {DEBUG && (
            <div className="debug">
              <div>
                backend: {detectorBackend ?? '—'} model:{' '}
                {detectorStats.model || '—'}
              </div>
              <div>
                infs: {detectorStats.inferences} fps:{' '}
                {detectorStats.fps.toFixed(1)} infer:{' '}
                {Math.round(detectorStats.inferMs)}ms
              </div>
              <div>
                maxScore: {detectorStats.maxScore.toFixed(3)} raw:{' '}
                {detectorStats.rawCount} kept: {detectorStats.keptCount}
              </div>
              {detectorStats.lastError && (
                <div className="debug-err">err: {detectorStats.lastError}</div>
              )}
            </div>
          )}
        </div>
      )}

      {phase === 'refining' && (
        <div className="screen running">
          <div
            className="analyzing-overlay"
            style={{ position: 'static', height: '100%' }}
          >
            <div className="spinner" />
            <div className="analyzing-text">解析中…</div>
            <div className="analyzing-sub">
              選択した {cart.size} 個の詳細を取得中
            </div>
          </div>
        </div>
      )}

      {phase === 'cart' && (
        <CartView
          cart={cart}
          refineError={refineError}
          onRemove={handleRemove}
          onReset={handleReset}
        />
      )}
    </div>
  );
}

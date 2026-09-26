/**
 * LLM モデルのタスク別定義。
 *
 * ★ このリポジトリでモデル名を書いてよいのはこのファイルだけ。★
 * refine-items.ts も live-token.ts も、ここから引く。
 *
 * 値の出所は Dustalk の中央台帳 `Platform/docs/models.yaml`。
 * 行末の `// role: xxx` がその写しであることを示していて、
 * `Platform/docs/scripts/check-models.py` が台帳とのズレを検出する。
 * **モデルを変えるときは台帳を先に直す**（規約は Platform/docs/decisions/002）。
 *
 * ファイル名が `_` で始まるのは Vercel の都合。api/ 直下のファイルは 1 つずつ
 * Function として公開されるので、handler を持たないこのファイルが
 * `/api/models` として叩かれないようにしている。
 *
 * 使い方:
 *     import { resolve } from './_models';
 *     const spec = resolve('refine');
 *     ai.models.generateContent({ model: spec.model, ... });
 *
 * 一覧:
 *     npm run models
 */

import { ThinkingLevel } from '@google/genai';
import { pathToFileURL } from 'node:url';

/**
 * LLM を呼ぶ用途。同じモデルを指していてもタスクは分ける。
 * 「音声だけ別モデルに落とす」を1行でできる状態を保つため（規約2）。
 */
export const LlmTask = {
  REFINE: 'refine', // 停止後の詳細化 (api/refine-items.ts)
  REFINE_FALLBACK: 'refine_fallback', // 上記が 503 のときの退避先
  LIVE: 'live', // 音声トークモード (api/live-token.ts)
  LIVE_FALLBACK: 'live_fallback', // Live の接続に失敗したときの退避先
} as const;

export type LlmTask = (typeof LlmTask)[keyof typeof LlmTask];

/** あるタスクが使うモデルの解決結果。 */
export type ModelSpec = {
  task: LlmTask;
  provider: string;
  model: string;
  /** 単価の単位。テキスト/画像は per Mtok、音声は per 分（規約4） */
  unit: 'mtok' | 'minutes';
  input: number;
  output: number;
  /** Gemini の thinking_level。null は未指定＝モデル任せ（規約5） */
  thinking?: ThinkingLevel | null;
};

// --- 台帳からの写し ---------------------------------------------------------
//
// 詳細化（切り抜き画像 → 構造化属性）。2026-08-15 に gemini-3.5-flash → 3.7-flash、
// 2026-09-26 に 3.7-flash → 3.8-flash（GA）へ移行した。3.6 / 3.7 / 3.8 Flash は単価が同じ。
// ⚠ 3.8 への移行時は実測していない。下の数字はすべて 3.7 以前のもの。
// 3.6 と 3.7 は単価が同じ（Flash 系全体の期間限定価格で、2027-01-01 に両方
// $1.50/$7.50 へ戻る）。**3.7 に移ったから安くなったのではない。**
//
// 720px の切り抜き5枚を本番プロンプトで実測（2026-08-15、現行単価）:
//   3.5-flash 未指定  in6439 out416 think1337  $0.0254  lat 8.9s ← 移行前
//   3.7-flash 未指定  in6439 out411 think762   $0.0092  lat 5.4s
//   3.7-flash low     in6439 out416 think0     $0.0064  lat 4.2s ← 採用
// 3条件とも5枚すべて同定に成功し、品目名の粒度も同等だった。
const REFINE = process.env.GEMINI_MODEL || 'gemini-3.8-flash'; // role: image-detect

// 503 UNAVAILABLE のときだけ使う退避先。**同格の GA モデル1本だけ**を置く。
// flash-lite 系を鎖に入れると、行が欠けた出力が静かに下流へ流れ、単価が安いので
// 請求でも気づけない（2026-08-15 決定。規約6）。
const REFINE_FALLBACK =
  process.env.GEMINI_FALLBACK_MODEL || 'gemini-3.6-flash'; // role: gemini-fallback

// 音声トークモード。2026-09-26 に gemini-3.1-flash-live-preview → gemini-3.8-live（GA）
// へ移行した。単価は 3.1 と同額（per 分）。音声は「統一方針の対象外」の整理のまま（規約7）。
// ⚠ 移行時に実機での会話品質・ツール呼び出しの挙動は実測していない。
const LIVE = process.env.GEMINI_LIVE_MODEL || 'gemini-3.8-live'; // role: voice-realtime

// Live の接続に失敗したときだけ使う。ephemeral token はモデル単位でロックされるので、
// live-token.ts はこのモデル用のトークンをもう1枚発行し、クライアントが順に試す。
// 別世代だが、Live API を持つモデル自体が少なく同格の代替が無い。
const LIVE_FALLBACK =
  process.env.GEMINI_LIVE_FALLBACK_MODEL ||
  'gemini-2.5-flash-native-audio-preview-12-2025'; // role: voice-realtime-fallback

/**
 * 詳細化の思考深度。
 *
 * 3.7 / 3.8 は思考トークンを出力として課金するので、未指定のままだと単価半減の効果を
 * 思考トークンが食う（上の実測で課金出力が 416 → 1,173 tok）。
 * `GEMINI_THINKING_LEVEL=auto` で未指定に戻せる。
 *
 * ⚠ 3.7 / 3.8 は `minimal` を受け付けない（400 INVALID_ARGUMENT）。low が下限。
 */
const REFINE_THINKING: ThinkingLevel | null = (() => {
  const v = (process.env.GEMINI_THINKING_LEVEL || 'low').toLowerCase();
  if (v === 'auto') return null;
  if (v === 'medium') return ThinkingLevel.MEDIUM;
  if (v === 'high') return ThinkingLevel.HIGH;
  return ThinkingLevel.LOW;
})();

const SPECS: Record<LlmTask, ModelSpec> = {
  refine: {
    task: 'refine',
    provider: 'gemini',
    model: REFINE,
    unit: 'mtok',
    input: 0.75,
    output: 3.75,
    thinking: REFINE_THINKING,
  },
  refine_fallback: {
    task: 'refine_fallback',
    provider: 'gemini',
    model: REFINE_FALLBACK,
    unit: 'mtok',
    input: 0.75,
    output: 3.75,
    thinking: REFINE_THINKING,
  },
  live: {
    task: 'live',
    provider: 'gemini',
    model: LIVE,
    unit: 'minutes',
    input: 0.005,
    output: 0.018,
  },
  live_fallback: {
    // 音声入出力の per Mtok 単価。3.1 Live と単位が違うので分換算はしていない。
    task: 'live_fallback',
    provider: 'gemini',
    model: LIVE_FALLBACK,
    unit: 'mtok',
    input: 3.0,
    output: 12.0,
  },
};

/**
 * タスクのモデルを解決する。
 *
 * 環境変数での上書きは各定数の側で受けている（規約3）。ローカルで別モデルを
 * 試すための口であって、本番の切り替え経路ではない。本番値はこのファイルの既定値。
 */
export function resolve(task: LlmTask): ModelSpec {
  return SPECS[task];
}

/** 全タスクの解決結果を返す（規約8）。 */
export function describeAll(): ModelSpec[] {
  return Object.values(LlmTask).map(resolve);
}

// `npm run models` で一覧を出す。Function からの import では走らない。
if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  console.log(
    `${'task'.padEnd(16)} ${'provider'.padEnd(9)} ${'model'.padEnd(46)} ${'price'.padEnd(22)} thinking`,
  );
  for (const s of describeAll()) {
    const unit = s.unit === 'mtok' ? 'per Mtok' : 'per 分';
    const price = `$${s.input}/$${s.output} ${unit}`;
    console.log(
      `${s.task.padEnd(16)} ${s.provider.padEnd(9)} ${s.model.padEnd(46)} ${price.padEnd(22)} ${s.thinking ?? '-'}`,
    );
  }
}

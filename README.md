# SmartCamera

スマホブラウザで動く、ライブ物体検知 + カゴ追加アプリ。

カメラ映像に対し **on-device の物体検出 (DEIMv2-N / ONNX)** を毎フレーム走らせて bbox を重ね、
**タップ**または**音声**で「捨てたい物」をカゴに入れる。停止すると選択した物体のスナップショットを
Gemini に送り、品目名・型番・メーカー等の構造化属性を得る。

仕様の詳細は [SPEC.md](./SPEC.md)、ローカル検出モデルの選定経緯は [MODEL_TRIALS.md](./MODEL_TRIALS.md) を参照。

## セットアップ

```bash
npm install
```

### 環境変数

`GEMINI_API_KEY` が必須。[Google AI Studio](https://aistudio.google.com/apikey) で発行。
リポジトリ直下に `.env`（ローカル）を作成するか、Vercel の Project Settings → Environment Variables に登録する。
キーはサーバ (Vercel Functions) 専用で、ブラウザには渡らない（Live は短命の ephemeral token だけを配る）。

**モデル名・単価・思考深度を書いてよいのは [`api/_models.ts`](./api/_models.ts) だけ**
（Dustalk 全体の規約。中央台帳 `Platform/docs/models.yaml` からの写しで、
`check-models.py` がズレを検出する）。`npm run models` で全タスクの解決結果が出る。

下記の環境変数はローカルで別モデルを試すための上書き口で、本番値はコードの既定値。

| 変数 | 既定値 | 用途 |
|---|---|---|
| `GEMINI_MODEL` | `gemini-3.7-flash` | 停止後の詳細化 (`/api/refine-items`) |
| `GEMINI_FALLBACK_MODEL` | `gemini-3.6-flash` | 上記が 503 のときの退避先（同世代・同単価） |
| `GEMINI_THINKING_LEVEL` | `low` | 思考深度。`auto` で未指定（モデル任せ）。**思考トークンは出力として課金される** |
| `GEMINI_LIVE_MODEL` | `gemini-3.1-flash-live-preview` | 音声トークモード (`/api/live-token`) |
| `GEMINI_LIVE_FALLBACK_MODEL` | `gemini-2.5-flash-native-audio-preview-12-2025` | Live の接続失敗時 |

### ローカル実行

API ルート (`/api/*`) を含めて動かすには Vercel CLI を使う:

```bash
npx vercel dev
```

UI だけなら `npm run dev` でも起動するが、詳細化と音声トークモードはエラーになる。

スマホ実機で確認するときは:
- `npx vercel dev --listen 0.0.0.0:3000` でネットワーク公開
- `getUserMedia` は HTTPS 必須なので、実機は Vercel preview URL に push して確認するのが早い

## ビルド & デプロイ

```bash
npm run build      # → dist/
npm run lint
```

Vercel に Git push すれば自動デプロイ（Framework: Vite、Functions: `api/`）。

## ファイル構成

```
api/
├── _models.ts             モデル定義（タスク別・単価・思考深度）。`_` 始まりなので Function 化されない
├── refine-items.ts        停止後の詳細化。切り抜き画像 → 構造化属性 (Gemini)
├── live-token.ts          Live の ephemeral token 発行。プロンプト/ツールをトークンに焼き込む
└── log.ts                 talk-mode 診断ログの受け口 (Vercel ログに [talklog] で出力)
public/models/
└── deimv2_n_640_uint8.onnx    ローカル検出モデル (INT8, ~4.4MB)
src/
├── App.tsx                live → analyzing → cart の状態遷移、タップ/音声の合流点
├── types.ts               Detection / SelectedEntry / Refined 型
├── useCamera.ts           getUserMedia + MediaStream 管理
├── useLocalDetector.ts    ONNX 推論ループ (document.hidden でポーズ)
├── yolo11.ts              モデルの前処理・後処理 (letterbox + score threshold)
├── localTracker.ts        ByteTrack 風トラッカー (速度予測で枠を補間)
├── detector.worker.ts     DEIMv2 推論を回す Web Worker
├── detectorConfig.ts      モデル URL / スコア閾値 (main と worker で共有)
├── captureSnapshot.ts     フレーム/切り抜きの JPEG 化 (長辺 720px, quality 0.8)
├── coords.ts              box_2d ↔ 正規化 xyxy の変換
├── live/
│   ├── useLiveSession.ts  Gemini Live セッション（音声 in/out + 1fps フレーム送信 + tool call）
│   ├── audio.ts           マイク録音 (16kHz リサンプル) と再生 (24kHz)
│   ├── cartTools.ts       add_to_cart 等のツールハンドラ
│   ├── DetectOverlay.tsx  bbox 描画 (Canvas 2D)
│   └── debugLog.ts        [live] 診断ログのバッファと /api/log への送信
├── cart/CartView.tsx      カート画面（属性表示 + サムネイル拡大）
└── ui/Button.tsx
```

## 技術メモ

- **検出はローカル、識別はクラウド**。DEIMv2-N は bbox の位置は良いが INT8 量子化で分類が崩れるため、
  UI ではラベルを出さず全 box を「物体」として扱い、品目名は Gemini に決めさせる
- ONNX は `onnxruntime-web` の **WASM 単スレ**で実行。iOS Safari の WebGPU は
  onnxruntime-web 側のメモリ問題が未解決なので使わない（[MODEL_TRIALS.md](./MODEL_TRIALS.md) 参照）
- **Live のセッション設定はサーバで ephemeral token に焼き込む**。クライアントが接続時に渡す
  `systemInstruction` / `tools` は無視されるため、`api/live-token.ts` が唯一の設定源
- トークンは**モデル単位でロック**されるので、フォールバック用に別トークンをもう1枚発行して
  クライアントが順に試す
- 音声で追加された物体は **Live に最後に送ったフレーム**を保持しておき、`refine-items` が返す
  `box_2d` で切り抜いてサムネイルにする（切り抜き判断は Live ではなく refine 側の責務）
- `gemini-3.7-flash` は Live API 非対応（native audio を持たない）ため、トークモードだけは
  3.1 Live 系に据え置いている

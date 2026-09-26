# SmartCamera 仕様書

スマホブラウザで動く、物体検知 + カゴ追加アプリ。**タップ + 音声のハイブリッド入力、ローカル検出 + Gemini 詳細化**:

1. ライブカメラに対し on-device で DEIMv2-N を毎フレーム (~1.5-3fps) 実行、bbox を重畳表示（ラベルは非表示 — 分類は信頼できないため）
2. **タップ入力**: 枠をタップ → その box をマージン付き (各辺 18%) で切り抜いた JPEG をスナップショットとして保持し、カゴに追加
3. **音声入力**: 🎙 で Gemini Live セッションを開始。「これ捨てたい」等の発話で Live モデルが `add_to_cart(name, box_2d, ...)` を呼ぶ。クライアントは **Live に最後に送信したフレーム**（モデルのカゴ入れ判断の根拠画像）を保持しており、モデルが返す `box_2d` でそのフレームを切り抜いてスナップショットにする。box_2d が無い/不正なら保持フレームをそのまま使う
4. 「停止」→ スナップショット群を 1 リクエストで Gemini に送り、各物体の構造化属性 (`name`, `description`, `modelNumber`, `manufacturer`, `yearOfManufacture`, `capacity`) を取得
5. カート画面で暫定ラベルを Gemini の `name` で置き換え、補助属性を 2 行目に、**切り抜きスナップショットをサムネイルとして表示（タップで拡大）**

オフラインや API 失敗時は暫定ラベル（タップ=「物体」、音声=発話された名称）で確定する (フォールバック)。

## ユーザーフロー

1. ページを開く → 「カメラ開始」をタップ
2. 背面カメラのライブプレビュー + bbox 重畳 (グレー点線) が出る
3. **タップ**: 物体をタップ → bbox が **青の実線** に変化、下部カゴ chip のカウントが増える
4. **音声**: 🎙 をタップ → Live 接続 →「このモニター捨てたい」「ペットボトル3本」等で追加。カゴ chip はタップ分と合算
5. 同じ物体を 2 度タップしても重複しない (LocalTracker の `instance_id` でデデュプ)。音声追加は負の instance_id で採番されタップ分と衝突しない
6. 最大 30 個まで（両経路合算）。上限到達時はタップ=注意表示、音声=モデルが capped を通知
7. 「停止」 → スピナー「解析中…」 (Gemini 5–15 秒)
8. カート画面: `具体名 × N` 表示。2行目に補助属性、下に各エントリの切り抜きサムネイル（タップで全画面拡大、背景タップで閉じる）
9. 「最初から」でリセット → idle 画面

ネットワーク失敗時はカート画面に到達し、各行は暫定ラベルのまま + 上部にエラーバナー表示。サムネイルは表示される。

## 技術スタック

| 領域 | 採用 |
|---|---|
| フロント | React 19 + Vite + TypeScript |
| ローカル推論 | DEIMv2-N INT8 ONNX @640² (ラベル非表示), `onnxruntime-web` WASM 単スレ |
| 詳細化 | `gemini-3.7-flash` (`@google/genai`, multimodal) |
| 音声トークモード | `gemini-3.1-flash-live-preview` (Gemini Live, native audio + 1fps 映像) |
| サーバ | Vercel Serverless Function (`/api/refine-items`, `/api/live-token`, `/api/log`) |
| カメラ | `getUserMedia` (`facingMode: 'environment'`) |
| 描画 | `<video>` + `<canvas>` オーバーレイ (Canvas 2D) |
| 物体追跡 | ByteTrack 風 2 段マッチ + alpha-beta 速度予測、描画毎フレーム外挿 (`LocalTracker`)。推論は Web Worker (`detector.worker.ts`) |
| 状態管理 | React state |
| デプロイ | Vercel (静的 SPA + Serverless API) |

## 動作環境

- iPhone 14 / iOS Safari 以降 (実機検証下限)
- Android Chrome (近代版)
- HTTPS 必須 (`getUserMedia` の制約)
- ネットワーク必須 (Gemini 詳細化)。オフライン時は YOLO ラベルで確定

## パラメータ初期値

| 項目 | 値 |
|---|---|
| ローカル推論 fps | デスクトップ ~3、iOS WebKit ~1.5 (`?fps=N` で上書き) |
| ローカル検出 入力解像度 | 640² (letterbox + 114-gray padding) |
| ローカル検出 スコア閾値 | 0.25 (`?conf=N` で上書き) |
| NMS | 不要 (DETR 系。postprocess 込みで top-300 が score 降順に返る) |
| スナップショット | 長辺 720px JPEG quality 0.8 |
| 切り抜きマージン | box 幅/高さの 18% (各辺、フレーム内クランプ) |
| 音声 box_2d 規約 | `[ymin, xmin, ymax, xmax]`、フレーム全体を 0-1000 とする正規化整数 (Gemini ネイティブ形式) |
| 選択上限 | 30 個 (タップ + 音声の合算) |
| Gemini モデル (詳細化) | `gemini-3.7-flash` → fallback `gemini-3.6-flash` (`GEMINI_MODEL` / `GEMINI_FALLBACK_MODEL`) |
| Gemini thinking | `low` を明示 (`GEMINI_THINKING_LEVEL`、`auto` で未指定に戻せる) |
| Gemini モデル (音声) | `gemini-3.1-flash-live-preview` → fallback `gemini-2.5-flash-native-audio-preview-12-2025` |
| Live フレーム送信 | 1 fps、長辺 720px JPEG (`frameIntervalMs`) |
| Vercel Function `maxDuration` | refine-items 60 秒 / live-token 15 秒 / log 10 秒 |

モデル名・単価・思考深度の定義は `api/_models.ts` に集約されている（Dustalk の中央台帳
`Platform/docs/models.yaml` からの写し。規約は同 `decisions/002-llm-model-definition.md`）。
上の環境変数はローカルで別モデルを試すための上書き口で、本番値はコードの既定値。
`npm run models` で全タスクの解決結果が出る。

## API 契約

### `POST /api/refine-items`

```jsonc
// Request
{
  "items": [
    {
      "id": 7,                              // クライアント側 instance_id (音声は負値)
      "yolo_label": "缶",                   // 暫定ラベル (タップ="物体"、音声=発話名)
      "image_b64": "<base64 JPEG>",         // マージン付き切り抜き (box 無しフォールバック時はフルフレーム)
      "image_mime": "image/jpeg",
      "bbox": [0.13, 0.13, 0.87, 0.87]     // 切り抜き画像内での物体位置 (inner box)、0-1 正規化 xyxy。
                                            // フルフレーム時は [0, 0, 1, 1]
    }
  ]
}

// Response (success)
{
  "items": [
    {
      "id": 7,
      "refined": {
        "name": "ペットボトル",             // 必須。カテゴリ表記
        "description": "500ml サイズの透明ボトル",
        "modelNumber": "",                  // 銘板から読めた場合のみ
        "manufacturer": "",                 // 無料引取候補カテゴリのみ
        "yearOfManufacture": "",
        "capacity": ""
      }
    }
  ]
}
```

`name` 以外の属性は最大限の best-effort、欠落する場合あり。Gemini が判定できなかったアイテムは items から省略される (フロントは暫定ラベルで継続表示)。

失敗時はステータス 500 + `{ "error": "..." }`、フロントは YOLO ラベルで確定 + 上部にエラーバナー。

## 描画スタイル

- 未追加ボックス: `strokeStyle = '#9CA3AF'`、`setLineDash([6, 4])`、線幅 2px
- 追加済ボックス: `strokeStyle = '#3B82F6'`、実線、線幅 3px
- タップフラッシュ: 青実線 4px が 700ms フェードアウト
- ラベル: ボックス上端に半透明黒背景 + 白文字

## 非機能要件

- スナップショット JPEG は **refine 後も保持**し、カート画面のサムネイル / 拡大表示に使う (マージン付き切り抜きなので 30 個でも数 MB 以内。サムネは `loading="lazy"` でデコード遅延)
- API キー (`GEMINI_API_KEY`) は Vercel 環境変数で管理、クライアントには露出しない
- Gemini は学習に使われない契約 (Google Cloud の AI 利用規約による)
- バックグラウンド化時に YOLO 推論を一時停止 (`useLocalDetector` の `document.hidden` ガード)

## 既知の割り切り

- 詳細化 (`/api/refine-items`) はキャンセル不可 (フロントから fetch を中断する手段は実装していない)
- LocalTracker は予測位置との IoU で照合し 3 秒の GC。3 秒以上見失った物体や、予測から大きく外れて再登場した物体は別 `instance_id` になる (= カートに重複追加され得る)
- カメラを速く動かすと YOLO の検出が安定しないので、撮影中はゆっくり動かす案内を継続
- Vercel リクエストボディは 4.5MB 上限。30 個 × ~80KB ≈ 2.4MB を想定 (720px / quality 0.8)、十分余裕
- 1 セッション 1 確定。連続スキャンは「最初から」で戻る

## コスト目安

単価は `gemini-3.7-flash` の導入価格 **$0.75 / $3.75 per Mtok**（2026-12-31 まで。2027-01-01 に倍）。

### 詳細化 `/api/refine-items`（2026-08-15 実測）

実測条件: 長辺 720px JPEG の切り抜き画像を N 枚、`thinking_level=low`（思考トークン 0）。

| 選択数 | 入力 tok | 出力 tok | 1リクエスト |
|---:|---:|---:|---:|
| 1 | 1,911 | 100 | $0.0018 |
| 3 | 4,150 | 273 | $0.0041 |
| 5 | 6,439 | 416 | **$0.0064** |
| 30（外挿） | 34,739 | 2,391 | $0.0350 |

固定オーバーヘッド（プロンプト + スキーマ）約 780 tok に、画像 1 枚あたり **入力 +1,132 tok / 出力 +79 tok** が乗る。

**思考深度の明示が効く**。同じ 5 枚で:

| 構成 | 入力 | 出力 | 思考 | 課金出力計 | コスト | レイテンシ |
|---|---:|---:|---:|---:|---:|---:|
| `3.5-flash` / 未指定（移行前） | 6,439 | 416 | 1,337 | 1,753 | $0.0254 | 8.9s |
| `3.7-flash` / 未指定 | 6,439 | 411 | 762 | 1,173 | $0.0092 | 5.4s |
| **`3.7-flash` / `low`（採用）** | 6,439 | 416 | **0** | 416 | **$0.0064** | 4.2s |

3 条件とも 5 枚すべて同定に成功し、`name` の粒度も同等。**思考トークンは出力として課金される**ため、
未指定のままモデルだけ上げると単価半減の効果を思考トークンが食う。

### 音声トークモード `/api/live-token`（試算）

`gemini-3.1-flash-live-preview`。映像 1fps は **258 tok/frame**（既定解像度）で text/image レート
$0.75/Mtok、音声は分単価（入力 $0.005/分・出力 $0.018/分）。

| 内訳 | 1 分あたり |
|---|---:|
| 映像 1fps（15,480 tok/分） | $0.0116 |
| 音声入力（常時） | $0.0050 |
| 音声出力（発話が全体の 1/3 と仮定） | $0.0060 |
| **計** | **$0.023/分** |

**映像が最大のコスト要因**（音声入力の 2.3 倍）。`mediaResolution: LOW` にすると 66 tok/frame まで落ち、
映像分が 1/4（$0.0030/分）になる。box_2d の判断は `refine-items` に移したので、Live 側の映像は
「何が写っているか」の把握だけに使われており、下げる余地がある（未検証）。

### セッション単位

| 使い方 | 概算 |
|---|---:|
| タップのみ・5 個選択 | **$0.0064**（約 ¥1） |
| 音声 2 分 + 5 個選択 | **$0.052**（約 ¥8） |
| 音声 5 分 + 30 個選択 | **$0.15**（約 ¥23） |

## ローカル検出モデル

**現行: DEIMv2-N (Intellindust AI Lab, Apache 2.0)、640² 入力、INT8 動的量子化 ~4.4MB** (`public/models/deimv2_n_640_uint8.onnx`)。採用経緯と試行履歴は `MODEL_TRIALS.md` を参照。DETR 系で NMS 不要、出力は postprocess 込み (`labels`/`boxes`/`scores`)。分類は信頼できないため UI ではラベル非表示・全 box「物体」扱いとし、識別は Gemini 詳細化に委ねる。

**2026-08-15 定点調査の結論**: nano帯 (≤4M params / INT8 ≤5MB) に DEIMv2-N を上回る選択肢はなし（ECDet は最小 S=10M で nano 帯なし、RF-DETR Nano は 30.5M で iOS WASM 圏外、YOLO26-N は AGPL）。iOS の WebGPU は onnxruntime-web のメモリ問題が**未修正のまま**（関連 issue は stale bot に自動クローズされただけ）なので WASM 単スレ維持。詳細は `MODEL_TRIALS.md` の「2026-08-15 定点調査」節。

---

以下は旧構成 (YOLOv11n-COCO) 時代の記録。閾値等の設計判断は現行にも引き継がれている。

ローカル推論は **Ultralytics 公式 `yolo11n.pt` (COCO-80)** をベースに **640² 入力**で ONNX export + INT8 動的量子化 (per-tensor) したものを使用。80 クラスとカバレッジは狭いが、クラスあたりの教師信号が密でリコールが高い。量子化後 ~2.9 MB。

設計判断: 「検出ラベルの細かさ」は **Gemini 詳細化 (`/api/refine-items`) に丸投げ**するため、YOLO 側は **bbox を確実に出すこと** を最重視。YOLO の表示ラベル ("ボトル" など) はユーザーがタップ対象を選ぶための粗いヒントで、確定ラベルは Gemini が返す `specific_name`。

候補比較:

| データセット | クラス数 | 出力テンソル (FP32) | 公式重み | リコール期待値 | 採否 |
|---|---|---|---|---|---|
| **COCO (採用)** | **80** | **~2.8 MB @640²** | ✅ `yolo11n.pt` | ◎ (密な教師) | ✅ |
| Open Images V7 | 601 | ~13 MB @512² | ✅ `yolov8n-oiv7.pt` | △ (信号薄) | 旧構成 |
| Objects365 | 365 | ~12 MB @640² | △ | ○ | 未試行 |
| LVIS | 1203 | ~40 MB @640² | △ community | △ ロングテール | iOS メモリ危険 |

入力解像度: COCO の出力テンソルは ~2.8MB と小さく、iOS の memory-pressure 余裕で **640² が可能**。OIV7 で 512² まで落としていた小物体リコールが復活。

### NMS は class-agnostic

`bowl` と `cup` が同じマグカップに同時発火するような重複を抑える目的で class-agnostic NMS (IoU > 0.5 で抑制) を採用 (`src/yolo11.ts:postprocess`)。

### SCORE_THRESHOLD = 0.15

低めに設定。false positive (誤分類された箱) はユーザーが選ばないだけなので問題にならず、逆に **検出漏れ (= タップできない)** が UX に直結する。`?conf=N` で実行時上書き可能。

### モデル成果物 (リポジトリ外の export 手順、再生成する場合)

```python
from ultralytics import YOLO
m = YOLO('yolo11n.pt')  # 公式重みを自動ダウンロード (~5.2 MB)
m.export(format='onnx', imgsz=640, opset=17, dynamic=False, simplify=True)

from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    'yolo11n.onnx',
    'yolo11n_coco_640_uint8.onnx',
    weight_type=QuantType.QUInt8,
    per_channel=False,  # per-tensor は onnxruntime-web WASM との相性が良い
)
```

ノートPC で数分、GPU 不要、fine-tuning 不要。

成果物:
- `public/models/yolo11n_coco_640_uint8.onnx` — 量子化済みモデル (~2.9 MB)
- `src/coco-labels.ts` — 80 クラスの日本語ラベル配列 + `labelOf(classId)` ヘルパー

### 検証要件

- iPhone 14 以降の実機で 5 分連続ライブ実行、タブキル無し
- 640² で iOS Safari がモデルロードに成功する (出力 ~2.8MB は memory-pressure 上限の十分内)
- 「ボトル」「コップ」「ノートPC」「スマホ」「リモコン」「本」など COCO 主要クラスで bbox が出ること
- 細かい物 (リップクリーム、USBケーブル等) は COCO に無いが、近隣の COCO クラスが箱を出してくれれば OK (タップ → Gemini が「リップクリーム」と返してくれる想定)

### ライセンス注意

Ultralytics YOLO は **AGPL-3.0**。社外公開する場合は商用ライセンス、または Apache-2.0 の代替 (YOLOE-S 等) への切替検討が必要。OIV7 自体のライセンスは Apache-2.0 + CC-BY-4.0 (再配布可)。

# ローカル検出モデル試行ログ

スマホブラウザ (iOS Safari / Android Chrome) WASM 単スレ環境で動かすローカル検出モデルの試行履歴。Gemini が後段でラベルを書き換えるので **絶対 mAP より bbox の recall + 数のバランス** を重視。

## 評価軸

各試行で以下を記録:

- **構成**: モデル名、入力解像度、量子化、ファイルサイズ
- **動作**: iOS Safari ロード可否、推論レイテンシ、平均 fps
- **検出品質**: 1 フレームあたりの kept box 数 (`?debug=1` の `kept`)、recall 体感、誤検出傾向
- **判断**: ✅ 採用 / ❌ 不採用 / ⚠️ 保留、理由

---

## 試行 #1 — YOLOv8n-OIV7 @ 512² (601 クラス)

- **期間**: 2026-05-17 〜 05-19
- **commit**: `731a32e` 〜 `a6114e6`
- **モデル**: `yolov8n-oiv7.pt` (Ultralytics 公式、Open Images V7 pretrain)
- **構成**: 入力 512²、INT8 動的量子化 per-tensor、~3.6MB、class-agnostic NMS、`SCORE_THRESHOLD=0.2`
- **狙い**: COCO 80 では語彙不足。OIV7 601 クラスでハサミ・USB ケーブル等の細かい物まで検出したい

### 結果

- **iOS ロード**: 当初 640² で "Load failed" (出力テンソル 20MB FP32 で memory-pressure 死)。512² に下げて回避
- **検出**: ❌ そもそも箱がほぼ出ない。`maxScore` が常時 0.1-0.2 付近で threshold 突破できず
- **原因**: 601 クラスに教師信号が薄まり、per-class sigmoid score が COCO 80 の半分以下になる

### 判断: ❌ 不採用

クラス数を増やしすぎると nano-tier モデルでは per-class confidence が崩れる。recall を上げるには threshold を 0.05 まで下げないといけないが、それだと誤検出が多すぎる。**「広い語彙 ⇆ 高い recall」のトレードオフは nano では選べない**。

---

## 試行 #2 — YOLOv11n-COCO @ 640² (80 クラス)

- **期間**: 2026-05-19 〜 (現在)
- **commit**: `c6418b2` (初回投入)、`7f0deb3` (FastSAM から復帰)
- **モデル**: `yolo11n.pt` (Ultralytics 公式、COCO pretrain)
- **構成**: 入力 640²、INT8 動的量子化 per-tensor、~2.9MB、class-agnostic NMS、`SCORE_THRESHOLD=0.15`
- **狙い**: 教師信号の密度を最大化。語彙は狭いが箱を確実に立てる

### 結果

- **iOS ロード**: ✅ 安定。出力テンソル ~2.8MB FP32 で余裕
- **検出**: ◯ COCO 主要クラス (人、車、椅子、ボトル、ノートPC、本、リモコン等) で箱が立つ
- **限界**: COCO 80 訓練範囲外 (USB ケーブル、書類、文房具、容器の蓋、リップ等) は **そもそも箱が立たない**

### 判断: ⚠️ ベースライン (現在のデフォルト)

語彙制約は本質的な限界。「COCO に居る物体しかタップできない」が許容できるなら ◯。Apache 2.0 ではなく AGPL-3.0 なのも将来課題。

---

## 試行 #3 — FastSAM-s @ 640² (class-agnostic)

- **期間**: 2026-05-20
- **commit**: `0e1fe96` (投入)、`b73d2ca` (4 段フィルタ追加)、`7f0deb3` (revert)
- **モデル**: `FastSAM-s.pt` (Ultralytics、YOLOv8s-seg を SA-1B 訓練)
- **構成**: 入力 640²、INT8 動的量子化、~11.8MB、単一クラス "object"、`SCORE_THRESHOLD=0.1` → 後に 0.25 + 面積フィルタ + containment フィルタ + top-15 cap
- **狙い**: 訓練範囲制約を捨てる。「画像内のあらゆる物体に箱」が SA-1B の教師タスク

### 結果

- **iOS ロード**: ⚠️ 12MB で動作したが境界線。詳細レイテンシ未計測
- **検出**: ❌ **箱が出すぎ**。SA-1B が「物体 + 部分 + 領域」を別々の正解として学習しているため、1 物体に複数箱が乱立
- **対策試行**: score threshold 0.25 化、NMS IoU 0.3 化、面積フィルタ 0.5%-70%、containment 80%、top-15 cap — それでも視認上多すぎ
- **根本原因**: SA-1B の "everything mask" 設計と「物体単位でタップ」の UX が噛み合わない

### 判断: ❌ 不採用

class-agnostic はアーキテクチャ的に正しいが、SA-1B 訓練のモデルは粒度が細かすぎる。**「object-level の単位で 1 物体 1 箱」が SA-1B からは復元しにくい**。

---

## 試行 #4 — DEIMv2-Pico @ 640² (実機テスト中)

- **期間**: 2026-05-21 〜
- **commit**: TBD (このコミット)
- **モデル**: DEIMv2-Pico (Intellindust AI Lab、CVPR 2025、Apache 2.0)
- **構成**:
  - backbone: **HGNetv2-Pico** (DINOv3 ではなく軽量 CNN backbone) ← 訂正
  - 入力 640²、INT8 動的量子化 per-tensor、**~2.3MB** (実測)
  - 200 queries (deploy postprocessor が上位 300 を返す)
  - SCORE_THRESHOLD = 0.25 (`?conf=N` で上書き可)
  - DETR set-prediction、**NMS 不要**

### Export 経緯

1. DEIMv2 リポジトリ `Intellindust-AI-Lab/DEIMv2` を clone
2. HuggingFace `Intellindust/DEIMv2_HGNetv2_PICO_COCO` から `model.safetensors` (6.3MB) を取得
3. safetensors → state_dict → 公式 `tools/deployment/export_onnx.py` に投入
4. **`load_state_dict` で missing keys 14 個** (`decoder.up`, `decoder.reg_scale`, `dec_bbox_head.1.*`, `dec_bbox_head.2.*`) → `strict=False` パッチで回避
   - `up` と `reg_scale` はモデルの `__init__` で `nn.Parameter` のデフォルト値 (0.5, 4.0) が設定済
   - `dec_bbox_head.1` と `.2` は `share_bbox_head=True` で `.0` と同一 Python オブジェクト = ロードは 1 回で足りる
5. opset 17 で export → FP32 ~6.4MB
6. `onnxruntime.quantization.quantize_dynamic` で UInt8 per-tensor → **2.3MB**

### 推論インターフェース

入力 (2 つ):
- `images`: float32 [1, 3, 640, 640]
- `orig_target_sizes`: int64 [1, 2] = [[640, 640]]

出力 (3 つ、postprocess 込み):
- `labels`: int64 [1, 300]、class ID
- `boxes`: float32 [1, 300, 4]、xyxy in input pixel space
- `scores`: float32 [1, 300]、score 降順

**JS 側 postprocess は score threshold だけ。** per-anchor argmax ループも NMS も letterbox 復元のみ。

### 開発機 (M4 Mac CPU) でのレイテンシ

ランダム入力で 22ms。iOS Safari WASM 単スレでは経験上 5-10 倍に落ちるので **100-200ms / フレーム** の見込み = **5-10fps**。これは現 YOLOv11n より速い可能性が高い (YOLOv11n は M4 CPU で ~30ms、iOS で ~600ms)。

### 結果

- **iOS ロード**: ✅ 成功 (2.3MB は余裕)
- **bbox 配置**: ◯ 物体に概ね正しく箱が立つ。FastSAM のような乱立はなく、YOLOv11n-COCO 並みのクリーンさ
- **classification**: ❌ 著しく劣化。例: **ポスターが「歯ブラシ」(COCO class 79) として分類**
- **原因推定**: 1.5M params + INT8 per-tensor 量子化で分類ヘッドが崩れる。DETR は attention 層が量子化感度高く、Pico 規模では誤分類が出やすい
- **mitigation**: COCO ラベルを UI に出さず **全 box を `"物体"` 固定表示** に変更 (commit TBD)。box 自体は使えるので Gemini 詳細化に流す前提で UX 損失ゼロ

### 判断: ⚠️ ラベル捨てて採用、ただし classification 改善余地あり

- box 配置は良好で、UI でラベル隠せば運用可能
- ただし classification の質を取り戻したいなら **DEIMv2-N (3.6M, AP 43.0)** に格上げが候補
- 一旦この構成 (Pico + ラベル隠し) でユーザー確認 → ダメなら次の試行へ

**追記 (2026-05-21)**: ユーザー要望で box 上のテキストラベルを完全に非表示化、live mode のカート chip も「選択中 ×N」に集約 (commit `19f9c39`)。

---

## 試行 #5 — DEIMv2-N @ 640²

- **期間**: 2026-05-21 〜
- **commit**: TBD (このコミット)
- **モデル**: DEIMv2-N (Intellindust AI Lab、CVPR 2025、Apache 2.0)
- **構成**:
  - backbone: HGNetv2-N
  - 入力 640²、INT8 動的量子化 per-tensor、**~4.4MB** (実測)
  - Pico と違って `share_bbox_head=False` + `gateway=True` で各層独立 = 表現力↑
  - I/O は Pico と同一: `images` + `orig_target_sizes` 入、`labels/boxes/scores` 出
  - SCORE_THRESHOLD = 0.25 維持
- **狙い**: Pico (AP 38.5) で box の質が惜しいときの格上げ。N は **AP 43.0** で +4.5、param 数 1.5M → 3.6M、INT8 サイズ 2.3MB → 4.4MB
- **コード変更**: 推論コード無変更、`modelUrl` の 1 文字差し替えのみ

### Export 経緯

Pico と同パイプライン:
1. HF `Intellindust/DEIMv2_HGNetv2_N_COCO` から safetensors (14MB) 取得
2. state_dict 化 → `tools/deployment/export_onnx.py` の strict=False パッチ版に投入
   - N は `share_bbox_head=False` なので tied weights 問題は無し、念のため strict=False で
3. FP32 ONNX 14.8MB → INT8 per-tensor 量子化 → **4.4MB**

開発機 (M4 Mac CPU) でランダム入力推論 ~27ms (Pico 22ms より 23% 遅い)。iOS WASM 単スレで 150-300ms / フレーム = 3-7fps 見込み。

### 結果

- **iOS ロード**: ✅ 成功 (4.4MB は余裕)
- **bbox 配置**: ✅ Pico より明確に改善。物体への当たりが良くなり、見落としも減
- **box 数**: 安定。1 物体 1 箱の傾向を維持 (FastSAM のような乱立なし)
- ラベルは UI で非表示 (`19f9c39` で設定済)、Gemini が後段で詳細化

### 判断: ✅ 採用 (current default)

box 品質を理由にした格上げ目的は達成。これ以上 (DEIMv2-S = 9.7M / ~10MB / AP 50.9) は box 品質伸びはあるがファイルサイズが iOS で OIV7-512² や FastSAM と同じ境界線に戻ってしまうので、**ここで一旦確定**。

更新が必要になる条件:
- 特定シーンで box の取りこぼしが目立つ → DEIMv2-S 検討
- iOS で 1fps を切るほど遅い → DEIMv2-Pico に降格 or DEIMv2-Atto (0.49M) 試行
- ライセンスや精度で要件変わる → YOLO26-N / RF-DETR 等再検討

---

## 試行 #6 — DEIMv2-S @ 640² (iOS メモリ上限の実機プローブ)

- **期間**: 2026-06-06 〜
- **commit**: TBD (このコミット、trial ブランチ)
- **モデル**: DEIMv2-S (Intellindust AI Lab、CVPR 2025、Apache 2.0)
- **構成**:
  - backbone: **DINOv3 ViT-Tiny (蒸留)** ← N までの HGNetv2 (CNN) と違い ViT 系
  - 入力 640²、INT8 動的量子化 per-tensor (QUInt8)、**~11.8MB** (実測)
  - I/O は N と同一: `images` + `orig_target_sizes` 入、`labels`/`boxes`/`scores` 出
  - SCORE_THRESHOLD = 0.25 維持、前処理も `/255` letterbox のまま (変更なし)
- **狙い**: 「~10MB 級の S 帯モデルが iOS Safari WASM で memory-pressure 死せず動くか」の**上限実測**が主目的。あわせて box 品質を N (AP 43.0) → S (AP 50.9) に格上げできるか確認。語彙は COCO-80 のまま。

### Export 経緯 (N と同パイプライン)

1. HF `Intellindust/DEIMv2_DINOv3_S_COCO` から `model.safetensors` (39.4MB / FP32) 取得
2. safetensors → `cfg.model.load_state_dict(state, strict=False)` で投入
   - missing 2 個 (`decoder.up`, `decoder.reg_scale`) のみ。これは `__init__` で default 値 (0.5, 4.0) が入る `nn.Parameter` なので問題なし (N と同じ挙動)
   - DINOv3 backbone の `weights_path: ./ckpts/vitt_distill.pt` はファイル不在で skip される (full ckpt を後段で load 済なので不要)
3. 公式 `export_onnx.py` 同等の deploy wrapper (model.deploy() + postprocessor.deploy()) で opset 17 export → FP32 ONNX **40.1MB**
4. `onnxruntime.quantization.quantize_dynamic` UInt8 per-tensor → **11.78MB**

### 量子化検証 (onnxruntime CPU, example.jpg)

FP32 と INT8、`/255only` と `mean/std` 正規化の 4 通りを比較:

- **INT8 は FP32 とほぼ同一の検出** (box 1-2px 差、max score 0.78→0.86)。**ViT backbone の量子化破壊なし**。
- **`/255only` (本番前処理) でも S は健全に動作**。mean/std 正規化との差は box 1px・検出数 30 vs 29 でほぼ同等 (正規化無しだとスコアが僅かに低いが threshold 0.25 で全部拾える)。
  → **yolo11.ts の前処理は変更不要**、純粋な modelUrl 差し替えで載る。

### 切替方法

`?model=s` で S を、無指定/`?model=n` で従来 N をロード (`src/yolo11.ts` の `pickModelUrl`)。**同一 iOS 端末で N/S を A/B 比較**してから採否を決める。

### 結果 (iOS 実機)

- TBD: ロード可否 (~11.8MB で memory-pressure 死しないか)
- TBD: 連続推論でのクラッシュ有無 (YOLO26 系の ~500 推論クラッシュ類似がないか数分回す)
- TBD: fps (N 比でどれだけ落ちるか)
- TBD: box 品質が N より体感で良くなるか

### 判断: ⚠️ 実機テスト中

---

## 2026-07 定点調査 — DEIMv2-N 維持を確定

ハイブリッド入力（タップ復活）にあたり box 表示モデルを再サーベイ（web 調査のみ、実機トライアルなし）。判断基準: bbox recall（タップ取りこぼし直結）/ INT8 ≤~5MB（iOS メモリ安全圏）/ Apache 2.0 系 / WASM 単スレ ≤~300ms/frame / 既存 I/O（`images`+`orig_target_sizes` → `labels/boxes/scores`、`src/yolo11.ts` の modelUrl 差し替えで済むか）。

2026-06 調査からの差分確認結果:

- **ECDet (EdgeCrafter)**: nano variant は依然なし（最小 S=10M/AP51.7、arXiv 2603.18739）。ONNX export スクリプト整備の兆しもなし
- **RF-DETR (ICLR 2026)**: Nano で AP48.0・Apache 2.0 と数字は良いが **params 30.5M のまま**（DINOv2 backbone）→ iOS WASM 圏外の判定変わらず
- **YOLO26**: 2026-01 リリースの N はあるが AGPL-3.0 のまま → ライセンス面で見送り継続
- **iOS WebGPU**: onnxruntime-web の WebKit26 メモリ暴走 (microsoft/onnxruntime#26827)、yolo26n が iOS26.3 で ~500 推論後クラッシュ (#27584) いずれも未解決 → WASM 単スレ維持が正解のまま

**結論: nano帯 (≤4M params / INT8 ≤5MB) に DEIMv2-N を上回る選択肢は 2026-07 時点でも存在しない。現行 DEIMv2-N を維持。** 更新条件（試行 #5 の記載）も変更なし。box recall に不満が出た場合の次手は DEIMv2-S (9.7M/AP50.9、export 手順既知) が最低リスク。
**その次手を実機で確かめているのが上の試行 #6**（`?model=s` で A/B）。

## 2026-08-15 定点調査 — DEIMv2-N 維持（変更なし）

Gemini 3.7 への移行に合わせて再サーベイ（web 調査のみ、実機トライアルなし）。判断基準は 2026-07 と同じ。

- **DEIMv2 (現行)**: HuggingFace の `Intellindust/DEIMv2_HGNetv2_{ATTO,FEMTO,PICO,N}_COCO` は **2025-10-29 以降更新なし**。N = 3.6M / AP 43.0 のまま。上流の最新ニュースは 2026-08-13 の「Intel® Geti への統合」で、モデル自体の更新ではない
- **ECDet (EdgeCrafter)**: HF に公開されているのは **ECDet/ECSeg/ECPose の S/M/L/X のみ**（最終更新 2026-03-30）。**nano 帯は 2026-08 時点でも存在しない**。ECDet-S は 10M/AP51.7 で DEIMv2-S・RT-DETRv4-S を上回るが、我々のサイズ制約の外
- **RF-DETR (ICLR 2026)**: Nano 30.5M / AP 48.0 のまま。D-FINE-Nano・LW-DETR-Tiny を 5AP 以上引き離すという主張は変わらないが、**params が桁で違う**ので iOS WASM 圏外の判定は不変
- **YOLO26-N**: AGPL-3.0 のまま。ライセンス面で見送り継続
- **iOS WebGPU**: 前回「未解決」とした 2 件は**いずれもクローズされたが、修正ではなく stale bot の自動クローズ**（#27584 が 2026-06-17、#26827 が 2026-07-18。最終コメントはどちらも bot で、修正コミットやリリースへの言及なし）。**直っていないので WASM 単スレ維持**。issue が閉じたことを「解決した」と読み違えないこと
- **onnxruntime-web**: latest は 1.27.0（手元は 1.25.1）。上げても上記の状況は変わらない

**結論: 2026-08-15 時点でも DEIMv2-N を置き換える候補はない。維持。** 次に見るタイミングは (a) DEIMv2 か EdgeCrafter が nano 帯の重みを出す、(b) onnxruntime の iOS WebGPU 問題に**実際の修正 PR** が入る、のいずれか。

## 2026-09-25 定点調査 + 構成改善（トラッカー / Worker）

モデル（web 調査のみ）:

- **YOLO-NAS**: 重みは Deci 独自の非商用ライセンス（本番利用不可）。Deci の NVIDIA 買収（2024-04）以降 super-gradients は保守停止 → **検討対象外**。YOLO 系は closed-set（学習クラスのみ）で、我々は検出ラベルを捨てているので語彙面の利点もない
- **RF-DETR**: Nano 30.5M / AP48.4 のまま（Apache 2.0）。iOS WASM 圏外の判定は不変
- **YOLO26-N / YOLOE-26**: AGPL-3.0 のまま
- **ECDet (EdgeCrafter)**: nano は依然なし（S/M/L/X）。**変化点: 公式リポジトリに `export/export_raw_onnx.py` が入った**（2026-07 時点では無かった）。試行 #6 で S 帯が iOS に載ると分かれば ECDet-S（10M / AP51.7）が DEIMv2-S（AP50.9）の次の候補
- **iOS WebGPU**: onnxruntime #26827 / #27584 に修正の形跡なし → WASM 単スレ維持

**結論: モデルは DEIMv2-N 維持。** 代わりにモデル以外のボトルネック（低 fps + 動きモデル無しトラッカー）に手を入れた:

- **推論を Web Worker 化**（`src/detector.worker.ts`）。これまでは iOS で 150–300ms の推論中にメインスレッドの rAF が止まっていた。フレームは `createImageBitmap` → transfer、前処理は OffscreenCanvas。onnxruntime-web が worker 側に移り main バンドルは 557kB → 485kB
- **トラッカーを ByteTrack 風に**（`src/localTracker.ts`）:
  - alpha-beta（固定ゲイン Kalman）で中心速度を推定する
  - 検出は**撮影時刻**で登録し、rAF ごとに現在時刻へ外挿して描画する（推論レイテンシ分の枠遅れも補正）
  - 2 段マッチ: score ≥ 0.25 で照合し、0.12–0.25 は既存トラックの延命のみに使う（`LOW_SCORE_FLOOR`）
  - 未マッチでも 900ms は表示を続け、3 秒は ID を保持して再捕捉する（旧: 1.5 秒）
  - シミュレーション（666ms 間隔、等速 0.05px/ms）で、描画位置の平均誤差は約 3.5px（旧方式は最後の検出を表示するだけなので平均約 17px 遅れ + 推論レイテンシ分）。BETA 0.5 は 5.3px で悪化したので 0.3 を採用
- `?debug=1` の HUD に model / 実効 fps / 推論 ms（EMA）を追加 → 試行 #6 の実機計測にそのまま使える
- Mac Chrome（バックグラウンドタブ）で、worker 経由の推論が COCO val 画像で動作することを確認（猫 2 / リモコン検出、164–206ms。バックグラウンドタブの throttle で試行 #5 の 27ms より遅い可能性あり、前面タブでの再計測が必要）

次の手:

1. iPhone で `?debug=1` と `?debug=1&model=s` を各数分回し、試行 #6 の TBD（ロード可否 / クラッシュ / fps / 枠の質）を埋める
2. S が載るなら ECDet-S を `export_raw_onnx.py` で ONNX 化し、I/O を合わせて試行 #7 とする
3. 中長期: Objects365 / LVIS を 1 クラスに統合し、自社撮影画像を加えて DEIMv2-N を class-agnostic に fine-tune する（COCO 外の品目の取りこぼし対策）

## 候補リスト (試行待ち)

優先度順:

1. **DEIMv2-Pico** ← 試行 #4 で実施中
2. **YOLO26-N** (Ultralytics 2025/10) — 既存 YOLOv11n の素直な後継、CPU 43% 速い、NMS-free、AGPL-3.0
3. ~~**DEIMv2-S** (9.71M / ~10MB)~~ ← 試行 #6 で実施中 (実測 INT8 11.8MB)
4. **YOLOv10n** (2024) — Apache 2.0、NMS-free、ライセンス避難先
5. **DEIMv2-Atto** (0.49M / ~1MB) — Pico で速度足りなければ降格

検討から外れたもの:

- **RF-DETR-Nano**: 名前は Nano だが 30.5M params → iOS WASM 圏外
- **YOLOv12n**: AGPL のまま、YOLO26 の方が新しく速い
- **YOLO-World / OWLv2 / Grounding DINO**: open-vocab 系は CLIP encoder 等で重すぎ
- **MobileSAM / EdgeSAM / TinySAM**: prompt-based なので "everything mode" が grid-sample で重い
- **LVIS-1203 系**: nano-tier で per-class 信号薄、OIV7 と同じ失敗パターン

# PSO駆動型動的結合ネットワーク研究の詳細設計

## 0. 実装サマリと利用手順

### 0.1 実装状況

- CIFAR-100 向けサブ専門家トレーナー（`program/train_sub.py`）を実装済み。層化サブセット抽出・データ拡張・softmax/logits 切替に対応。
- PSO ゲーティング最適化パイプライン（`program/pso_train.py`／`program/train_pipeline.py`）を整備し、平均結合行列を履歴 (`avg_gating`) として記録。
- ゲーティング遷移のネットワークアニメーション（`program/visualize_gating.py`）で GIF / MP4 出力とインタラクティブ確認が可能。
- 評価指標は精度・冗長性・複雑度・滑らかさを JSON で出力。FLOPs・モジュラリティの自動算出と継続学習シナリオは未実装。

注：現行の最小実装として MNIST サブ NN 学習（`program/train_sub_nn.py`）が含まれます。まずはこれで環境確認が可能。

### 0.2 セットアップと基本コマンド（uv 前提）

```bash
# 依存関係の同期（.venv が作成されます）
uv sync

# MNIST サブ NN の学習（ルートでモジュール実行）
uv run -m program.train_sub_nn

# もしくは、インストール済みエントリポイント（uv sync 後）
uv run train-sub-nn

# 出力: program/models/test_model/ 配下（デフォルト）
```

注意:
- Python バージョンは TensorFlow の互換性上、`>=3.10,<3.13` を使用します。本リポジトリでは `.python-version` を `3.12` に固定しています。uv のメッセージに従って `uv python pin 3.12` を実行しても同等です。
- コマンドは必ずリポジトリのルートで実行してください（`program/` 直下からファイル実行すると相対インポートが失敗します）。

## 1. 背景と動機

### 1.1 静的な深層学習の課題

現代の深層学習は固定ネットワーク（CNN / Transformer）上で勾配法によりパラメータを更新する枠組みが主流である。しかし固定構造では全入力に対して同一の計算グラフを走らせるため計算効率・適応性に限界があり、継続学習では破滅的忘却を招きやすい。入力依存で幅や経路を変える動的ニューラルネットワークはこの硬直性を克服する。特にゲーティング機構が重要であり、入力から各モジュールの活性度を調整して実行幅を選択することで、効率・表現力・解釈性を両立できる（例: [pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov)）。

### 1.2 Mixture-of-Experts (MoE) とゲーティングモデル

MoE は複数専門家 (experts) と入力依存で重み付けするゲート (gating model) から成る。ゲートはソフトマックス重みベクトルを出力し、専門家出力の線形和で最終予測を形成する（参考: [machinelearningmastery.com](https://machinelearningmastery.com)）。分割統治により入力領域ごとに最適な専門家を活性化できる。本研究では重みベクトルを一般化し、専門家間相互作用を表現する結合行列生成へ拡張する。

### 1.3 Particle Swarm Optimization (PSO)

PSO は粒子群が速度と位置を更新し適応度を最大化する進化的最適化。勾配不要で非微分な罰則項を含む目的関数に適用しやすい。全体最良 (gbest) / 局所 (lbest) トポロジ選択で探索–収束バランスを調整可能。[pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov) などに詳細。ゲーティング重みを連続ベクトルとして扱い、PSO で結合構造を探索する。

## 2. 研究目的

本研究では、動的な計算グラフを自律的に形成できるニューラルアーキテクチャを構築し、勾配に頼らない PSO によってゲーティング構造を最適化することを目標とする。達成すべきポイントは以下のとおり。

- **動的結合機構の構築**：複数の専門家ネットワークと、入力に基づいて専門家間の結合行列を生成するゲーティングネットワークを設計する。ゲーティング重みは PSO で探索する。
- **計算効率と適応性の向上**：従来の固定アーキテクチャや MoE と比較し、入力ごとに結合を変えることで分類性能や継続学習性能の改善を狙う。
- **創発的構造の分析**：PSO で獲得された結合構造を可視化し、協調・抑制パターンの観察と解釈を行う。

## 3. 理論的基盤と設計

### 3.1 動的ニューラルネットワークとゲーティング

動的ネットワークは入力や時間に応じて構造や重みを変える。動的幅を実現するゲーティング機構では、ゲート関数が入力から各専門家の活性レベルを出力し、必要な専門家のみを選択する（例：[pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov)）。MoE のゲーティングモデルは入力に応じて専門家の貢献度を決定し、その出力（ソフトマックス確率）が専門家の重みとして用いられる（[machinelearningmastery.com](https://machinelearningmastery.com)）。

本研究ではゲーティングネットワークを結合行列生成関数として一般化する。入力 \(x\) に対してゲーティングネットワーク \(g_\theta\) が専門家数 \(N\) の結合行列 \(C_x \in \mathbb{R}^{N \times N}\) を生成し、専門家同士の情報交換強度を決定する。従来の MoE は一次元の重みベクトルを出力して専門家の線形和を取るが、本研究では時間方向の再帰的情報交換を導入し、専門家間の協調・競合関係を学習する。結合行列は非対称を許容し、長期的な情報流を表現できるようにする。

### 3.2 PSO によるゲーティング関数の最適化

粒子 \(i\) の速度・位置更新は次式で与えられる：

\[
\mathbf{v}_i(t+1) = \omega \mathbf{v}_i(t) + c_1 r_1 (p_i - \mathbf{x}_i(t)) + c_2 r_2 (g - \mathbf{x}_i(t)), \quad
\mathbf{x}_i(t+1) = \mathbf{x}_i(t) + \mathbf{v}_i(t+1)
\]

ここで \(\omega\) は慣性重み、\(c_1, c_2\) は認知係数・社会係数、\(r_1, r_2\) は一様乱数ベクトルである。位置更新は勾配を参照せずデータ駆動的に進むため、非微分な目的関数にも適用できる。

適応度関数の例：

\[
F(\theta) = \alpha \cdot \text{Acc} - \beta \cdot \text{Redundancy} - \gamma \cdot \text{Complexity} - \delta \cdot \text{Smoothness}
\]

- **Acc**：分類精度（トップ1／トップ5）。
- **Redundancy**：専門家出力の相関（多様性確保）。
- **Complexity**：結合行列の密度や計算量に対する罰則。
- **Smoothness**：再帰ステップ間の結合行列変化量に対する罰則。

慣性・係数 (\(\omega, c_1, c_2\)) やトポロジ設定 (gbest / lbest) を調整することで探索と収束のバランスを取る。

### 3.3 モデルアーキテクチャ

#### 3.3.1 専門家ネットワーク (Sub-NN)

- CIFAR-100 を扱う小規模 CNN を採用。Conv-BN-ReLU ブロック＋Global Average Pooling＋全結合層で 100 クラスのロジットを出力。
- Dropout やラベルスムージングで過学習を抑制し、各専門家は異なるサブセット（10〜20%）で事前学習する。
- Softmax 版モデルを事前学習に用い、PSO ではロジット版を読み込む。

#### 3.3.2 ゲーティングネットワーク

- 入力：画像または専門家の中間特徴（32×32×3 など）。
- 出力：\(N \times N\) の結合行列 \(C_x\)。行ごとにソフトマックス正規化し、遷移確率として解釈する。自己結合を許容するかは実験で検証する。
- 構造：軽量 CNN または MLP で特徴を圧縮し、最終層で \(N \times N\) の線形出力を得る。

#### 3.3.3 再帰的情報交換

- 初期の専門家出力ベクトル \(O^{(0)} \in \mathbb{R}^{N \times C}\) に対し、結合行列による線形変換と正規化を \(T\) ステップ反復（\(O^{(t+1)} = \phi(C_x O^{(t)})\)）。
- \(\phi\) は恒等写像＋LayerNorm 等の安定化処理を想定。最終出力は各専門家ベクトルを平均または重み付き和で集約する。

### 3.4 タスクと実験設計

**基礎性能検証 (CIFAR-100)**

- データセット：CIFAR-100（100 クラス、32×32×3）。
- 専門家数：8〜10。
- 事前学習：各専門家を 10〜20% の層化サブセットで学習し、トップ1精度 35〜55% 程度に抑える。
- PSO 設定：粒子数 30〜50、反復回数 300〜500、慣性重みを 0.9 → 0.4 へ線形減衰、認知・社会係数を 1.5〜2.0。

### 4.3 可視化と解析

- PSO 反復ごとに平均結合行列をヒートマップ／ネットワークグラフで可視化し、結合パターンの変化を観察。
- 専門家の出力を UMAP や t-SNE で低次元化し、クラス分布と多様性を分析。
- 適応度や粒子多様性（速度分散・位置分散）の推移を追跡し、探索と収束の挙動を評価。

## 5. 理論的検討と予想

- **動的構造適応と計算効率**：動的幅モデルは入力ごとに計算リソースを調整でき、不要な専門家を抑制しながら精度を維持できる。動的深さ／幅／ルーティングは結合行列生成で統一的に表現可能。
- **PSO の利点**：非微分な罰則項を含む目的関数でも最適化でき、初期値や学習率への感度が小さい。局所トポロジや慣性調整で多峰性空間でも探索が安定する。
- **専門家多様性の重要性**：冗長性ペナルティやサブセット分割、ラベルスムージングにより専門家間の多様性を確保し、意味のある結合学習を促進する。
- **期待される成果**：固定アーキテクチャや標準 MoE よりも高い精度を目指し、継続学習・クラス不均衡への耐性向上を期待。結合行列の可視化で入力依存構造の解釈性を得る。

## 6. 実施計画

1. **準備**：CIFAR-100 前処理、専門家ネットワーク／ゲーティングネットワーク実装、PSO ベクトル化ユーティリティ、実験ログ基盤を整備。
2. **事前学習実験**：専門家ごとの精度と冗長性を測定し、サブセット割合やモデルサイズを調整。
3. **PSO 実験**：粒子数・慣性・係数・トポロジのグリッド探索を行い、目的関数重み (\(\alpha, \beta, \gamma, \delta\)) の感度分析を実施。
4. **比較実験**：固定 CNN、平均アンサンブル、勾配による MoE、動的深さモデルと性能・計算量・メモリ使用量を比較。
5. **継続学習実験**：タスク追加時の性能変化と結合行列変動を分析し、破滅的忘却の軽減効果を評価。
6. **論文化・可視化**：結合ヒートマップ、PSO ダイナミクス、専門家関係図、性能比較グラフなどを整理し、成果をまとめる。

### 6.1 実験手順

以下は CIFAR-100 を用いた標準的な一連の実験フローです。各ステップは再現性確保のためシード指定を推奨します。

1. 環境準備
	- Python 仮想環境作成・有効化。
	- 依存をインストール: `pip install -e .`。
	- GPU 利用確認 (`tensorflow` が GPU を認識するか `tf.config.list_physical_devices('GPU')`)。

2. 専門家事前学習 (Sub-Experts)
	- CIFAR-100 をロードし、`subset_pool_fraction` でクラスごとにプールするサンプルを決定（例: 0.2）。
	- `program/train_sub.py` を利用し層化分割されたインデックス集合で各専門家を学習。
	- Softmax 版を学習し、その後ログイット版へ重みをコピー (`logits.weights.h5`)。

	```bash
		 # 8 専門家を層化 20% プールで 3 epoch 学習
			 # uv のモジュール実行（推奨）
			 uv run -m program.train_sub \
	  --num-experts 8 \
	  --subset-pool-fraction 0.2 \
	  --epochs 3 \
	  --batch-size 128 \
	  --label-smoothing 0.1 \
	  --learning-rate 1e-3 \
	  --subset-seed 42 \
	  --output-root ./models/cifar_sub_experts

			 # エントリポイント（uv sync 後）
			 uv run train-sub \
				 --num-experts 8 \
				 --subset-pool-fraction 0.2 \
				 --epochs 3 \
				 --batch-size 128 \
				 --label-smoothing 0.1 \
				 --learning-rate 1e-3 \
				 --subset-seed 42 \
				 --output-root ./models/cifar_sub_experts
	```

	生成物: `models/cifar_sub_experts/expert_XX/` 下に `history.json`, `softmax.weights.h5`, `logits.weights.h5`, `train_indices.npy`。

3. PSO 用評価セット構築
	- 全専門家の `logits.weights.h5` をロード。
	- CIFAR-100 全体（またはランダムサブセット `--sample-count`）に対する各専門家のロジットを前計算しメモリ保持。
	- これが `program/pso_train.py` 内 `precompute_expert_logits(...)` により `expert_logits` テンソルとして構築される。

4. PSO によるゲーティング最適化
	- ゲーティングモデル（入力 → N×N 行列）を構築し `WeightAdapter` でフラット化。
	- 粒子初期化：モデル重みに微小ノイズを加えたベクトル。
	- 各反復で FitnessEvaluator が適応度 (Acc, Redundancy, Complexity, Smoothness) を計算。
	- 慣性重みは線形減衰（`config.PSO_INERTIA_MAX`→`MIN`）。

		 ```bash
		 # uv のモジュール実行（推奨）
		 uv run -m program.pso_train \
	  --experts ./models/cifar_sub_experts \
	  --num-experts 8 \
	  --sample-count 4096 \
	  --batch-size 128 \
	  --hidden-units 384 \
	  --lr 1e-3 \
	  --seed 123 \
	  --iterations 120 \
	  --particles 24 \
	  --output ./models/pso_gating

		 # エントリポイント（uv sync 後）
		 uv run pso-train \
			 --experts ./models/cifar_sub_experts \
			 --num-experts 8 \
			 --sample-count 4096 \
			 --batch-size 128 \
			 --hidden-units 384 \
			 --lr 1e-3 \
			 --seed 123 \
			 --iterations 120 \
			 --particles 24 \
			 --output ./models/pso_gating
	```

	出力: `pso_history.json` (各 iteration のスナップショット), `gating_weights.npy` (最良ベクトル), `fitness.json`。

5. 再帰的結合評価
	- `config.PSO_RECURRENT_STEPS` 回だけ専門家ロジット集合をゲーティング行列で更新し平均化。
	- 再帰ステップごとの差分ノルムが Smoothness に寄与。

6. 可視化
	- 平均ゲーティング行列の時系列をネットワークアニメーション化：
		 ```bash
		 uv run -m program.visualize_gating \
	  --history ./models/pso_gating/pso_history.json \
	  --out ./models/pso_gating/gating_anim.gif \
	  --threshold 0.02 \
	  --fps 6 --show

		 # エントリポイント
		 uv run viz-gating \
			 --history ./models/pso_gating/pso_history.json \
			 --out ./models/pso_gating/gating_anim.gif \
			 --threshold 0.02 \
			 --fps 6 --show
	```
	- `avg_gating` のヒートマップ生成は任意のノートブックで `plt.imshow(np.array(entry['avg_gating']))`。

7. アブレーション実験
	- 冗長性ペナルティ除去: `config.FITNESS_BETA = 0.0`。
	- 再帰無効化: `config.PSO_RECURRENT_STEPS = 1`。
	- トポロジ差分: 現状 gbest のみ → lbest を導入するには `ParticleSwarmOptimizer` を近傍集合更新へ拡張。
	- 勾配ベース比較: ゲーティングモデルを通常の `model.fit` で学習（専用スクリプト要追加）。

8. 継続学習シナリオ（将来拡張）
	- タスク分割: クラス集合を時間順に分割 (例: 20 クラスずつ)。
	- 各タスクで専門家を追加／既存専門家固定。
	- 追加クラスに対して再度 PSO 最適化し旧タスク精度変化を測定。
	- 指標: 旧タスク保持率 = 新タスク後の旧タスク精度 / 旧タスク学習直後の精度。

9. 結果整理
	- 主要指標とハイパーパラメータを `results.csv` などに集約。
	- 冗長性 vs 精度 のトレードオフ曲線、慣性減衰カーブ、ゲート行列モジュラリティ（未実装）を可視化。

### 6.2 実験成果物一覧

| 種類 | パス例 | 内容 |
|------|--------|------|
| 専門家重み | `models/cifar_sub_experts/expert_00/logits.weights.h5` | ロジット版学習済み CNN |
| 学習履歴 | `models/cifar_sub_experts/expert_00/history.json` | epoch ごとの loss / acc |
| PSO 履歴 | `models/pso_gating/pso_history.json` | iteration, inertia, best_score, avg_gating |
| 最終ゲート | `models/pso_gating/gating_weights.npy` | 最良ゲーティング重みベクトル |
| 適応度 | `models/pso_gating/fitness.json` | 最終 score 各指標値 |
| 可視化 | `models/pso_gating/gating_anim.gif` | ゲート行列時系列アニメーション |

### 6.3 再現性メモ

- 乱数シード: 専門家分割 `--subset-seed`, PSO 最適化 `--seed` を明示記録。
- バージョン: TensorFlow/Keras のバージョンと GPU ドライバ情報を `ENVIRONMENT.md`（未作成）へ追加推奨。
- 計測: 追加で FLOPs や推論レイテンシを取得する場合は `tf.profiler` / `thop` 相当の補助スクリプトを別途作成予定。

## 7. まとめ

本研究は、専門家間の結合行列を PSO によって自己組織的に最適化する動的ニューラルネットワークを提案する。動的幅と再帰的結合により、入力やタスクに応じてネットワーク構造を適応的に変化させることが可能となる。PSO は非微分な罰則を含む複雑な目的関数にも対応でき、勾配法では得にくい大域探索能力を活かせる。計算効率・適応性・容量・解釈性の利点を備え、継続学習や資源制約環境への応用が期待される。
### 8. 詳細比較実験計画（PSO-DCN vs MoE 他）

本節では PSO 駆動型動的結合ネットワーク (PSO-DCN) を従来手法と比較する体系的プランを整理する。最終目標は CIFAR-10/100 で Top-1 ~80% 近傍を達成しつつ冗長性・計算量の削減と構造解釈性を示すこと。

#### 8.1 対象ベースライン
- 固定単一 CNN (Single)
- 平均アンサンブル (Avg-Ensemble)
- 勾配学習 MoE (Grad-MoE: ソフトマックス重み学習)
- ランダムゲート (Random-Gate: 専門家重み一様/ディリクレ)
- Stacking (Stacking: 専門家出力をメタ学習器 MLP/SVM に入力)
- PSO-DCN (本提案) + アブレーション（冗長性ペナルティ無効 / 再帰無効 / 慣性固定）

#### 8.2 指標
- 精度: Top-1 / Top-5
- 冗長性: 専門家ロジット相関平均 (低いほど多様)
- 複雑度: 有効結合密度・推定 FLOPs (後追い実装)
- 滑らかさ: 再帰ステップ間の行列差分ノルム
- メモリ: 学習済み重み総サイズ (MB)
- レイテンシ (任意): 100 サンプル平均推論時間

#### 8.3 期待成果物
| 種類 | ファイル | 内容 |
|------|----------|------|
| ベースライン結果 | `results/baselines.json` | 各手法指標まとめ |
| PSO 進行 | `results/pso/fitness.csv` | iteration ごとの各項目 |
| 結合履歴 | `results/pso/C_history.npy` | 平均結合行列時系列 |
| アブレーション | `results/ablation/*.json` | ペナルティ除去等の比較 |
| 構造可視化 | `results/fig/gating_anim.gif` | 時系列ネットワーク |

#### 8.4 コマンド例
```bash
# 固定単一 CNN 学習（Class-IL リプレイ無し）
uv run single-cnn \
	--dataset cifar10 \
	--epochs 60 \
	--seed 1 \
	--output ./results/single

# 平均アンサンブル学習 (N 専門家再利用)
uv run ensemble-eval \
	--experts ./models/cifar_sub_experts \
	--num-experts 8 \
	--dataset cifar100 \
	--output ./results/ensemble

# 勾配 MoE 学習
uv run moe-train \
	--experts ./models/cifar_sub_experts \
	--num-experts 8 \
	--dataset cifar100 \
	--hidden-units 384 \
	--epochs 40 \
	--output ./results/moe

# ランダムゲート評価
uv run random-gate \
	--experts ./models/cifar_sub_experts \
	--num-experts 8 \
	--dataset cifar100 \
	--sample-count 4096 \
	--trials 10 \
	--output ./results/random

# Stacking メタ学習
uv run stacking-fit \
	--experts ./models/cifar_sub_experts \
	--num-experts 8 \
	--dataset cifar100 \
	--epochs 30 \
	--output ./results/stacking

# PSO-DCN 最適化
uv run pso-train --experts ./models/cifar_sub_experts --iterations 300 --particles 40 --output ./results/pso

# アブレーション例（冗長性ペナルティ 0）
FITNESS_BETA=0 uv run pso-train --experts ./models/cifar_sub_experts --iterations 300 --particles 40 --output ./results/pso_no_redund

# ベースライン結果の集約
uv run aggregate-results ./results/single ./results/ensemble ./results/moe ./results/random ./results/stacking --output ./results/baselines.json
```

#### 8.5 再現性事項
- 乱数: データ分割 / PSO / MoE / Stacking で個別 seed を記録。
- バージョン: TensorFlow/Keras, CUDA, cuDNN を冒頭に出力するユーティリティ追加予定。
- 設定スナップショット: 主要ハイパーパラメータを JSON 化 (`run_config.json`)。

#### 8.6 今後の拡張
- CIFAR-10 Split Class-IL 継続学習シナリオ
- FLOPs / モジュラリティ計測ユーティリティ
- lbest PSO トポロジ実装と比較
- 温度付きゲート (Softmax 温度調整) による sparsity 制御

> 注: 単一 CNN 用スクリプトのみ未整備。その他は README 記載の CLI で実行可能です。
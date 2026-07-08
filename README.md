<h1 align="center">rss-sentiment-arbitrage</h1>

<p align="center">
    <strong>指定したニュースメディアのRSSから、 ユーザーの関心領域外（フィルターバブル外）のニュースを自動抽出し、 LLM(Gemini API)を用いて大学生向けに構造化・平易化した カスタムRSSフィードを生成・配信するシステム</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3100/">
    <img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Python 3.10">
  </a>
  <a href="https://github.com/ph-cookie/rss-sentiment-arbitrage/actions">
    <img src="https://github.com/ph-cookie/rss-sentiment-arbitrage/actions/workflows/update-rss.yml/badge.svg" alt="Build Status">
  </a>
  <a href="https://ph-cookie.github.io/rss-sentiment-arbitrage/">
    <img src="https://img.shields.io/badge/Hostedon-GitHubPages-brightgreen.svg" alt="GitHub Pages">
  </a>
  <a href="https://aistudio.google.com/">
    <img src="https://img.shields.io/badge/Poweredby-Gemini3.1FlashLite-orange.svg" alt="Gemini API">
  </a>
  <a href="https://huggingface.co/intfloat/multilingual-e5-small">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97HuggingFace-multilingual--e5--small-yellow.svg" alt="Hugging Face">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
</p>

## 1. システム概要

現代の情報収集におけるフィルターバブル（推薦アルゴリズムによる関心の偏り）を打破するための、逆フィルタリング型ニュース配信システムです。

事前定義した複数の関心テキスト（興味クラスタ）と各記事のコサイン類似度を計算し、最も類似度が低い（関心外である）記事のみを抽出。最新のLLM（Google Search Grounding連携）を用いつつ、興味を惹かれるタイトルへのリライトと構造化された解説を自動生成し、GitHub Pages経由で新たなRSSフィード（XML）及びインデックスページとして配信します。

> [!WARNING]
> 本システムが生成するAIによる解説や要約は、ユーザーに専門外の分野に対する興味・関心を持たせるための「導入」および「補完」を目的としています。
>事実関係についてはGoogle Search Grounding等を用いて精度向上を図っていますが、AI特有のハルシネーションや最新情報の反映漏れが含まれる可能性があります。
> **正確な事実関係や詳細については、必ずフィード内のリンクから本来のニュース記事（元記事）を通読することを前提に、ご確認ください。**

## 2. システムフロー

<details>
<summary style="color: #666; font-size: 0.9em; cursor: pointer;">
🔍 <b>クリックしてシステムフロー図を表示</b>
</summary>

```mermaid
graph TD
    A[GitHub Actions 定期実行] --> B[キャッシュ読込: 既読URL & ストック記事]
    B --> C[複数RSSフィードのパース]
    C --> D{既読URL or 有料限定か}
    D -- Yes --> E[スキップ / 除外ログ出力]
    D -- No --> F[新規・無料記事の蓄積]
    
    F --> G{新規記事あり?}
    G -- No ---> K
    G -- Yes --> H[SentenceTransformerによる一括ベクトル化]
    
    H --> I{複数興味クラスタとの\n最大類似度 < THRESHOLD}
    I -- 対象あり --> J[関心外ニュースの抽出]
    I -- 0件 --> J2[類似度下位3件を強制抽出]
    
    J --> L[LLM + Google Search Groundingによる\n構造化生成 & タイトルリライト]
    J2 --> L
    L -- "API制限等: tenacityで自動再試行 & 5秒スロットリング" --> M[MIME動的判定 / 余白最適化]
    
    M --> K[新規記事と過去記事を結合し、最新30件を抽出]
    K --> N[キャッシュ保存: 最新の既読URL & 最新30件の記事データ]
    N --> O[feedgenによるカスタムRSSファイル生成]
    O --> Q[index.html 案内ページの自動生成]
    Q --> P[GitHub Pages への自動デプロイ]
```

</details>

## 3. 主な機能

* 興味クラスタによる高精度な逆フィルタリング    
    単一のテキストによる意味の希釈化を防ぐため、興味関心を複数のクラスタ（配列）として定義。各記事に対し全クラスタとの類似度を計算し、最大類似度ベースで判定することで高精度に「関心外」を特定します。

* Google Search Groundingによる時事情報の補完    
    Gemini APIの検索連携機能を有効化。AIの事前学習知識に頼らず、最新の時事情報や普遍的な構造的背景を正確に補完した解説を生成します。

* 常時ストック方式によるフィード維持    
    actions/cache を利用し、処理済みURL（最大500件）と生成済み記事データをJSONで保存。重複処理を防ぎつつ、常に最新30件の記事を維持して出力するため、新規記事が0件のタイミングでも過去の記事が消滅せず安定した配信を実現します。

* APIレートリミット対策（堅牢なエラーハンドリング）    
    無料API枠（15 RPM等）の超過を防止するため、LLM生成部に5秒間の事前スロットリング（待機処理）を導入。さらに tenacity を用いた指数的バックオフにより、一時的な通信エラーにも最大5回まで自動再試行します。万が一の生成失敗時も元記事の要約でフォールバックし、システムを止めません。

* RSS表示の最適化    
    記事概要に元ソース名と類似度スコアを明記。画像URLから動的にMIMEタイプを判定する堅牢なenclosure対応や、インラインCSSによるHTML余白最適化を行っています。


## 4. テクニカルスタック

* 言語: Python 3.10
* LLM SDK: google-genai (最新仕様 / Types対応)
* 生成モデル: gemini-3.1-flash-lite (Google Search Grounding有効)
* 埋め込みモデル: sentence-transformers (intfloat/multilingual-e5-small)
* リトライ制御: tenacity
* RSS生成: feedgen
* インフラ: GitHub Actions (CI/CD), GitHub Pages (静的ホスティング)

## 5. リポジトリ構成

* `main.py`: RSS取得、一括ベクトル化、LLM生成、XML出力までの全パイプラインを管理するメインスクリプト
* `requirements.txt`: 2026年現在の依存パッケージ一覧
* `.github/workflows/generate-rss.yml`: 定期実行およびキャッシュ制御を行うGitHub Actions定義ファイル

## 6. セットアップ手順

1. **リポジトリの準備**
   本リポジトリを自身のGitHubアカウントにクローン、またはフォークして作成する。

2. **各種APIキー・トークンの取得**
   * [Google AI Studio](https://aistudio.google.com/) から Gemini API キーを取得。
   * [Hugging Face](https://huggingface.co/settings/tokens) から Access Token (Read権限) を取得。

3. **GitHub Secrets の設定**
   GitHubリポジトリの `Settings` > `Secrets and variables` > `Actions` に、以下の環境変数を登録する。
   * `API_KEY1`: 取得したGemini APIキー
   * `HF_TOKEN1`: 取得したHugging Faceトークン

4. **GitHub Pages の有効化**
   GitHubリポジトリの `Settings` > `Pages` にて、Build and deployment の Source を「GitHub Actions」に設定する。

5. **ソースコードのカスタマイズ**
   `main.py` 内の以下の定数を、自身の情報収集目的に応じて変更する。
   * `SOURCE_RSS_URLS`: 取得対象とするニュースメディアのRSS URLリスト
   * `INTEREST_TEXT`: 自身の現在の興味（これと離れた記事が抽出される）
   * `THRESHOLD`: 類似度の閾値（デフォルト: 0.821）
   * `EXCLUDE_KEYWORDS`: 有料記事などを弾くための除外キーワード群

## 7. 利用方法

GitHub Actionsの実行が正常に完了すると、GitHub Pages環境へ自動デプロイされ、以下のURLに案内ページ（index.html）が生成されます。

`https://[GitHubユーザー名].github.io/[リポジトリ名]/`

同ディレクトリ内の rss.xml を、FeedlyやNetNewsWireなどの任意のRSSリーダーアプリに登録して購読してください。

## LICENSE

MIT

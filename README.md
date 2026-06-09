# rss-sentiment-arbitrage

指定したニュースメディアのRSSから、ユーザーの**関心領域外（フィルターバブル外）のニュース**を自動抽出し、Gemini APIを用いて大学生向けに構造化・平易化したカスタムRSSフィードを生成・配信するシステム。

## 1. システム概要

現代の情報収集におけるフィルターバブル（推薦アルゴリズムによる関心の偏り）を打破するための、逆フィルタリング型ニュース配信システム。
事前定義した関心テキストと各記事のコサイン類似度を計算し、類似度が低い（関心外である）記事のみを抽出。LLMを用いて、興味を惹かれるタイトルへのリライト、および構造化された詳細な解説（概要、背景・要因、社会・読者への影響）を自動生成し、GitHub Pages経由で新たなRSSフィード（XML）として配信する。

## 2. システムフロー

```mermaid
graph LR
    %% スタイリング（見やすさのための微調整）
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px;
    classDef highlight fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;

    subgraph "Phase 1: 収集とフィルタリング"
        A([GitHub Actions<br>定期実行]) --> B[(キャッシュ読込<br>既読URL/前回記事)]
        B --> C[複数RSS<br>パース]
        C --> D{既読 or<br>有料?}
        D -- Yes --> E[スキップ]
        D -- No --> F[新規・無料記事<br>蓄積]
    end

    subgraph "Phase 2: ベクトル化と抽出"
        F --> G{新規記事<br>あり?}
        G -- Yes --> H[SentenceTransformer<br>一括ベクトル化]
        H --> I{興味類似度<br>< THRESH}
        I -- 対象あり --> J[関心外ニュース<br>抽出]
        I -- 0件 --> J2[下位3件を<br>強制抽出]
    end

    subgraph "Phase 3: 記事生成と配信"
        J & J2 --> L[Gemini 3 Flash<br>構造化生成/リライト]
        L -. API制限時は<br>自動再試行 .-> M[MIME動的判定<br>余白最適化]
        
        %% 新規がない場合のバイパスルート
        G -- No ----> K
        
        M --> K[新規記事 ＋<br>前回記事を結合]
        K --> N[(キャッシュ保存<br>URL/今回記事)]
        N --> O[feedgen<br>RSS生成]
        O --> P([GitHub Pages<br>自動デプロイ]):::highlight
    end
```

## 3. 主な機能

* ローカルキャッシュと履歴の2世代保持: actions/cache を利用し、処理済みURLのリスト（最大150件）と「前回実行時に生成した記事データ」をJSONとして保持。APIの無駄な再実行（重複処理）を防ぎつつ、RSSフィードには「今回＋前回」の2世代分の記事が出力され、記事の読み逃しを防止する。
* バッチベクトル化と逆フィルタリング: SentenceTransformer (multilingual-e5-small) を用い、蓄積した全記事のベクトル計算を一括実行。設定した閾値以下の「関心外」ニュースのみを高効率に抽出。
* フォールバック抽出: 閾値未満の記事が0件の場合、相対的に類似度が低い下位3件を強制抽出し、フィードの出力を保証。
* AIタイトルリライト & 構造化平易化: gemini-3-flash を用い、大学生の興味を惹く30文字以内のタイトルへ刷新。本文を「概要」「背景・要因」「社会・読者への影響」の3項目に構造化し、400〜600文字で深掘り解説。
* 堅牢なエラーハンドリング: tenacity による指数的バックオフをLLM生成部に実装。APIのレートリミット（429エラー）発生時も最大5回まで自動再試行。
* 運用監視ログの強化: 有料記事の除外理由（ヒットしたキーワード）やフィードごとの処理件数を詳細に構造化ログ出力。
* RSS表示の最適化: 各記事の概要欄（description）に元のソースサイト名と類似度スコアを明記。「概要なし」によるリーダー側の表示崩れを防止し、インラインCSSによる余白調整、画像の埋め込み（enclosure対応）を最適化。

## 4. テクニカルスタック

* 言語: Python 3.10
* LLM SDK: google-genai (最新仕様)
* 生成モデル: gemini-3-flash
* 埋め込みモデル: sentence-transformers (intfloat/multilingual-e5-small)
* リトライ制御: tenacity
* RSS生成: feedgen
* インフラ: GitHub Actions (CI/CD), GitHub Pages (ホスティング)

## 5. リポジトリ構成

* main.py: RSS取得、一括ベクトル化、LLM生成、XML出力までの全パイプラインを管理するメインスクリプト
* requirements.txt: 2026年現在の最新安定版パッケージ定義
* .github/workflows/generate-rss.yml: 毎日指定時刻（日本時間 6:00, 12:00, 19:00）に動作する実行定義ファイル

## 6. セットアップ手順

### 1. リポジトリの準備

本リポジトリを自身のGitHubアカウントにクローン、またはフォークして作成する。

### 2. 各種APIキー・トークンの取得

* Google AI Studio から Gemini API キーを取得。
* Hugging Face から Access Token (Read権限) を取得。

### 3. GitHub Secrets の設定

GitHubリポジトリの Settings > Secrets and variables > Actions に、以下の環境変数を正確に登録する。

* API_KEY1: 取得したGemini APIキー
* HF_TOKEN1: 取得したHugging Faceトークン

### 4. GitHub Pages の有効化

GitHubリポジトリの Settings > Pages にて、Build and deployment の Source を「GitHub Actions」に設定する。

### 5. ソースコードのカスタマイズ

main.py 内の以下の定数を、自身の情報収集目的に応じて変更する。

* SOURCE_RSS_URLS: 取得対象とするニュースメディアのRSS URLリスト
* EXCLUDE_KEYWORDS: 有料記事などを弾くための除外キーワード群
* INTEREST_TEXT: 自身の現在の興味（これと離れた記事が抽出される）
* THRESHOLD: 類似度の閾値（デフォルト: 0.821）

## 7. 利用方法

GitHub Actionsの実行が正常に完了すると、GitHub Pages環境へ自動デプロイされる。
生成された以下のURLを、FeedlyやNetNewsWireなどの任意のRSSリーダーアプリに登録して購読する。

https://[GitHubユーザー名].github.io/[リポジトリ名]/rss.xml
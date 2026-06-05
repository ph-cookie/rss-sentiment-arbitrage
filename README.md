# rss-sentiment-arbitrage

## 問題意識

現代の情報収集は推薦アルゴリズムに依存しており、自身の既存の関心領域に情報が偏る「フィルターバブル」が避けられない。

例えば、YouTubeやInstagramは、閲覧履歴やいいね、滞在時間から関連コンテンツを優先表示する。そのため、興味のある情報だけが強化され、視野が狭まる。

これはニュースにおいても同様と言える。特に金融やコンサルティングなどのビジネス領域においては、専門外のマクロ経済や異業種の動向を見落とすことは致命的な機会損失となる。

しかし、関心のない難解なニュースを日常的に読み込むことは学習コストが高く、継続が困難である。本プロジェクトは、類似度計算を用いて意図的に「関心外」のニュースを抽出し、LLMを用いて「ユーザーの既存の関心事（IT、プログラミング、芸術など）のアナロジー（類推）」に変換して解説する。これにより、認知的負荷を下げつつ情報の死角を補完する。

## システム概要

指定したニュースメディアのRSSを取得し、事前の設定テキストとのコサイン類似度が低い（＝興味がない）記事のみをフィルタリングする。抽出された記事はGemini APIによって平易な解説文に再構築され、新たなカスタムRSS(XML)としてGitHub Pages経由で配信される。

## 主な機能

* ローカルAIモデル(multilingual-e5-small)による記事のベクトル化と類似度計算
* コサイン類似度を用いた「関心外」ニュースの逆フィルタリング
* Gemini APIを利用した、指定領域の概念に基づくアナロジー解説の自動生成
* GitHub Actionsによる完全自動運用と静的ホスティング

## アーキテクチャ

* 言語: Python 3.10
* ベクトル化: SentenceTransformers (Hugging Face)
* LLM: Google Gemini API (gemini-1.5-flash)
* インフラ: GitHub Actions (定期実行), GitHub Pages (RSSホスティング)

## ファイル構成

* main.py: メインの処理スクリプト（RSS取得、ベクトル化、テキスト生成、XML出力）
* requirements.txt: 依存パッケージ一覧
* .github/workflows/update-rss.yml: GitHub Actionsの自動実行定義ファイル

## セットアップ手順

1. 本リポジトリを自身のGitHubアカウントに作成（またはフォーク）。
2. Google AI StudioでGemini APIキーを取得。
3. Hugging FaceでAccess Token (Read)を取得。
4. GitHubリポジトリの Settings > Secrets and variables > Actions にて以下を登録。
* GEMINI_API_KEY: 取得したGeminiのAPIキー
* HF_TOKEN: 取得したHugging Faceのトークン


5. GitHubの Settings > Pages にて、Build and deployment の Source を「GitHub Actions」に設定。
6. main.py 内の SOURCE_RSS_URL と INTEREST_TEXT を自身の環境や目的に合わせて変更。

## 購読方法

Actionsの実行完了後、GitHub Pagesで公開された `https://[ユーザー名].github.io/[リポジトリ名]/rss.xml` を、FeedlyやNetNewsWireなどの任意のRSSリーダーアプリに登録する。
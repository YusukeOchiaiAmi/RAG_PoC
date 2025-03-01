# RAG PoC（Retrieval-Augmented Generation Proof of Concept）

このプロジェクトは、RAG（Retrieval-Augmented Generation）システムの概念実証（PoC）です。ローカルで動作するLLMを使用して、与えられたドキュメントに基づいた質問応答システムを実装しています。

## 参考にさせていただいた記事

CPUだけでローカルLLM(GPU不使用) - Qiita (<https://qiita.com/kansou/items/58aff8b89ee999306141>)

## 使用するモデル

Llama-3-ELYZA-JP-8B-gguf (Llama-3-ELYZA-JP-8BのGGUF形式、5GB程度)

## 環境要件

- **GPU不要**：このシステムはCPUのみで動作します
- Python（v3.12で動作確認済み）
- 必要なパッケージ：
  - langchain および関連パッケージ
  - llama-cpp-python (<https://abetlen.github.io/llama-cpp-python/whl/cpu>)

## セットアップ手順

### パッケージのインストール

```bash
pip install langchain langchain-community langchain-core faiss-cpu sentence-transformers

# バイナリホイール形式のパッケージを優先してインストール
pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### モデルのダウンロード

models/配下に以下のURLからモデルをダウンロード

<https://huggingface.co/mmnga/Llama-3-ELYZA-JP-8B-gguf/blob/main/Llama-3-ELYZA-JP-8B-Q4_K_M.gguf>

### ドキュメントのベクトル化

1. documents配下に UTF-8 形式のテキストファイルを追加
2. vector.pyを実行して、vectorstore/配下にベクトルデータベースが保存される

(参照するドキュメントがなくても、RAGなしのLLMモードとして動作します)

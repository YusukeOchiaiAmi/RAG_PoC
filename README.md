# RAG PoC（Retrieval-Augmented Generation Proof of Concept）

このプロジェクトは、RAG（Retrieval-Augmented Generation）システムの概念実証（PoC）です。ローカルで動作するLLMを使用して、与えられたドキュメントに基づいた質問応答システムを実装しています。

## 環境要件

- Python 3.12（動作確認済み）
- 必要パッケージ：
  - langchain および関連パッケージ
  - llama-cpp-python (<https://abetlen.github.io/llama-cpp-python/whl/cpu>)
- 使用するモデル:
  - Llama-3-ELYZA-JP-8B-gguf (Llama-3-ELYZA-JP-8BのGGUF形式)

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

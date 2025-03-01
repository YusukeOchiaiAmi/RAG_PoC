from llama_cpp import Llama
from langchain.prompts import PromptTemplate
from vector import load_vectorstore

def setup_rag_system(model_path, vectorstore_path="vectorstore"):
    """RAGシステムのセットアップ"""
    # Llamaモデルを初期化
    llm = Llama(
        model_path=model_path,
        chat_format="llama-3",
        n_ctx=2048,  # より長いコンテキストを扱えるように拡張
    )

    # ベクトルストアをロード
    vectorstore = load_vectorstore(persist_directory=vectorstore_path)
    if vectorstore is None:
        print("ベクトルストアがロードできません。RAGなしのLLMモードで続行します。")

        # RAGなしのLLM応答関数を定義
        def llm_query(query):
            # Llamaモデルを使用して回答を生成
            messages = [
                {"role": "system", "content": "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。"},
                {"role": "user", "content": query}
            ]

            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=1024,
            )

            return {
                "answer": response["choices"][0]["message"]["content"],
                "source_documents": []
            }

        return llm_query

    # リトリーバーを作成
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # プロンプトテンプレートの作成
    template = """あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。
以下の情報を参考にして、ユーザーの質問に回答してください。
情報に関連がない質問には「関連する情報がありません」と答えてください。

参考情報:
{context}

質問: {question}
"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # RAG関数を定義
    def rag_query(query):
        # 関連するドキュメントを検索
        documents = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in documents])

        # Llamaモデルを使用して回答を生成
        messages = [
            {"role": "system", "content": "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。"},
            {"role": "user", "content": prompt.format(context=context, question=query)}
        ]

        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=1024,
        )

        return {
            "answer": response["choices"][0]["message"]["content"],
            "source_documents": documents
        }

    return rag_query

def main():
    # モデルパス
    model_path = "models/Llama-3-ELYZA-JP-8B-Q4_K_M.gguf"

    # RAGシステムをセットアップ
    query_function = setup_rag_system(model_path)
    if query_function is None:
        print("システムのセットアップに失敗しました。")
        return

    # モードを表示
    is_rag_mode = len(query_function("test")["source_documents"]) > 0
    mode = "RAG" if is_rag_mode else "LLMのみ"
    print(f"{mode}モードで準備ができました。質問を入力してください。")

    # 対話ループ
    while True:
        query = input("\n質問（終了するには 'exit'）: ")
        if query.lower() == 'exit':
            break

        # 回答を生成
        result = query_function(query)

        # 回答を表示
        print("\n回答:")
        print(result["answer"])

        # 参照ドキュメントの表示(存在する場合)
        if result["source_documents"]:
            print("\n参照ドキュメント:")
            for i, doc in enumerate(result["source_documents"]):
                print(f"\nドキュメント {i+1}:")
                print(f"内容: {doc.page_content[:200]}...")
                if 'source' in doc.metadata:
                    print(f"ソース: {doc.metadata['source']}")

if __name__ == "__main__":
    main()
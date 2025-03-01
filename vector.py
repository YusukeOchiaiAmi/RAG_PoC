import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def load_documents(directory_path):
    """指定ディレクトリからドキュメントを読み込み、分割する"""
    print(f"ドキュメントの読み込みを開始: {directory_path}")
    loader = DirectoryLoader(directory_path, glob="**/*")
    documents = loader.load()
    print(f"読み込み完了: {len(documents)}ファイル")

    # ドキュメントを分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"分割完了: {len(chunks)}チャンク")

    return chunks

def create_vectorstore(chunks, embeddings_model_name="intfloat/multilingual-e5-small", persist_directory="vectorstore"):
    """ドキュメントチャンクをベクトル化してFAISSベクトルストアに保存"""
    # 埋め込みモデルの初期化
    print(f"埋め込みモデルの初期化: {embeddings_model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    # ベクトルストアの作成
    print(f"ベクトルストアの作成を開始")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # ベクトルストアの保存
    print(f"ベクトルストアを保存: {persist_directory}")
    vectorstore.save_local(persist_directory)

    return vectorstore

def load_vectorstore(embeddings_model_name="intfloat/multilingual-e5-small", persist_directory="vectorstore"):
    """保存されたベクトルストアをロードする"""
    # 埋め込みモデルの初期化
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    # ベクトルストアのロード
    print(f"ベクトルストアをロード: {persist_directory}")
    if os.path.exists(persist_directory):
        vectorstore = FAISS.load_local(persist_directory, embeddings)
        return vectorstore
    else:
        print(f"ベクトルストアが見つかりません: {persist_directory}")
        return None

def directory_has_content(directory_path):
    """ディレクトリ内にファイルやフォルダが存在するかをチェック"""
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        return False

    # ディレクトリ内のすべての項目をリスト
    contents = os.listdir(directory_path)

    # 隠しファイル（.で始まるファイル）を除外してカウント
    non_hidden_contents = [item for item in contents if not item.startswith('.')]

    return len(non_hidden_contents) > 0

def main():
    # 設定
    documents_dir = "documents"
    vectorstore_dir = "vectorstore"
    embeddings_model = "intfloat/multilingual-e5-small"

    # ドキュメントディレクトリのコンテンツをチェック
    if not os.path.exists(documents_dir):
        print(f"ドキュメントディレクトリが見つかりません: {documents_dir}")
        print("documentsディレクトリを作成して、テキストファイルを配置してください。")
        return
    elif not directory_has_content(documents_dir):
        print(f"ドキュメントディレクトリにファイルがありません: {documents_dir}")
        print("documentsディレクトリにテキストファイルを配置してください。")
        return

    # ドキュメントの読み込みと分割
    chunks = load_documents(documents_dir)

    if len(chunks) == 0:
        print("処理可能なドキュメントが見つかりませんでした。")
        print("テキストファイル(.txt)を配置してください。")
        return

    # ベクトルストアの作成と保存
    _ = create_vectorstore(
        chunks,
        embeddings_model_name=embeddings_model,
        persist_directory=vectorstore_dir
    )

    print("ベクトル化が完了しました。")
    print(f"チャンク数: {len(chunks)}")
    print(f"ベクトルストアが保存されました: {vectorstore_dir}")

if __name__ == "__main__":
    main()
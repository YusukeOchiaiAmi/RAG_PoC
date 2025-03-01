import os
from typing import List, Optional
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

def load_documents(directory_path: str) -> List[Document]:
    """指定ディレクトリからドキュメントを読み込み、分割する

    Args:
        directory_path: ドキュメントを読み込むディレクトリのパス

    Returns:
        分割されたドキュメントチャンクのリスト
    """
    print(f"ドキュメントの読み込みを開始: {directory_path}")
    # TextLoaderを使用してテキストファイルのみを読み込む
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt",  # .txtファイルのみをターゲット
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}  # UTF-8で
    )
    documents: List[Document] = loader.load()
    print(f"読み込み完了: {len(documents)}ファイル")

    # ドキュメントを分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks: List[Document] = text_splitter.split_documents(documents)
    print(f"分割完了: {len(chunks)}チャンク")

    return chunks

def create_vectorstore(
    chunks: List[Document],
    embeddings_model_name: str = "intfloat/multilingual-e5-small",
    persist_directory: str = "vectorstore"
) -> FAISS:
    """ドキュメントチャンクをベクトル化してFAISSベクトルストアに保存

    Args:
        chunks: ドキュメントチャンクのリスト
        embeddings_model_name: 埋め込みモデルの名前
        persist_directory: ベクトルストアを保存するディレクトリ

    Returns:
        作成されたFAISSベクトルストア
    """
    # 埋め込みモデルの初期化
    print(f"埋め込みモデルの初期化: {embeddings_model_name}")
    embeddings: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    # ベクトルストアの作成
    print(f"ベクトルストアの作成を開始")
    vectorstore: FAISS = FAISS.from_documents(chunks, embeddings)

    # ベクトルストアの保存
    print(f"ベクトルストアを保存: {persist_directory}")
    vectorstore.save_local(persist_directory)

    return vectorstore

def load_vectorstore(
    embeddings_model_name: str = "intfloat/multilingual-e5-small",
    persist_directory: str = "vectorstore"
) -> Optional[FAISS]:
    """保存されたベクトルストアをロードする

    Args:
        embeddings_model_name: 埋め込みモデルの名前
        persist_directory: ベクトルストアが保存されているディレクトリ

    Returns:
        ロードされたFAISSベクトルストア、失敗した場合はNone
    """
    # 埋め込みモデルの初期化
    embeddings: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    # ベクトルストアのロード
    print(f"ベクトルストアをロード: {persist_directory}")
    if os.path.exists(persist_directory):
        vectorstore: FAISS = FAISS.load_local(
            persist_directory,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vectorstore
    else:
        print(f"ベクトルストアが見つかりません: {persist_directory}")
        return None

def directory_has_content(directory_path: str) -> bool:
    """ディレクトリ内にファイルやフォルダが存在するかをチェック

    Args:
        directory_path: チェックするディレクトリパス

    Returns:
        ディレクトリに非隠しファイルが含まれている場合はTrue
    """
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        return False

    # ディレクトリ内のすべての項目をリスト
    contents: List[str] = os.listdir(directory_path)

    # 隠しファイル（.で始まるファイル）を除外してカウント
    non_hidden_contents: List[str] = [item for item in contents if not item.startswith('.')]

    return len(non_hidden_contents) > 0

def main() -> None:
    """メイン関数: ドキュメントの読み込み、ベクトル化、保存を行う"""
    # 設定
    documents_dir: str = "documents"
    vectorstore_dir: str = "vectorstore"
    embeddings_model: str = "intfloat/multilingual-e5-small"

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
    chunks: List[Document] = load_documents(documents_dir)

    if len(chunks) == 0:
        print("処理可能なドキュメントが見つかりませんでした。")
        print("テキストファイル(.txt)を配置してください。")
        return

    # ベクトルストアの作成と保存
    _: FAISS = create_vectorstore(
        chunks,
        embeddings_model_name=embeddings_model,
        persist_directory=vectorstore_dir
    )

    print("ベクトル化が完了しました。")
    print(f"チャンク数: {len(chunks)}")
    print(f"ベクトルストアが保存されました: {vectorstore_dir}")

if __name__ == "__main__":
    main()
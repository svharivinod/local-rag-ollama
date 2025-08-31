# ingest.py
from pathlib import Path
from chromadb import PersistentClient
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

# ——— absolute paths & settings ———
ROOT = Path(__file__).parent.resolve()
PDF_PATH = ROOT / "data" / "attentionpaper.pdf"
PERSIST_DIR = str((ROOT / "chroma_db").resolve())
COLLECTION = "rag"
SETTINGS = Settings(anonymized_telemetry=False)

def main():
    assert PDF_PATH.is_file(), f"PDF not found at {PDF_PATH}"
    print("📁 Project root:", ROOT)
    print("📄 PDF path:", PDF_PATH)
    print("💾 DB path:", PERSIST_DIR)

    print("📄 Loading PDF…")
    pages = PyPDFLoader(str(PDF_PATH)).load()
    print(f"   Loaded {len(pages)} pages")

    print("✂️  Splitting into chunks…")
    # You can tweak these if recall feels low: try 800/120
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    print(f"   Created {len(chunks)} chunks")

    print("🔢 Embedding with Ollama: mxbai-embed-large …")
    embedder = OllamaEmbeddings(model="mxbai-embed-large:latest")

    print(f"🧠 Building Chroma index at {PERSIST_DIR} (collection='{COLLECTION}') …")
    client = PersistentClient(path=PERSIST_DIR, settings=SETTINGS)

    # Create/populate the SAME collection explicitly
    _ = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        client=client,
        collection_name=COLLECTION,
    )

    # Verify persisted count from disk
    db = Chroma(
        client=client,
        collection_name=COLLECTION,
        embedding_function=embedder,
    )
    print(f"🧪 Persisted docs in '{COLLECTION}': {db._collection.count()}")
    print("✅ Ingestion complete.")

if __name__ == "__main__":
    main()

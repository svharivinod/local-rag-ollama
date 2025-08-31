# 🧠 Local RAG with Ollama + Chroma

This project demonstrates a simple **Retrieval-Augmented Generation (RAG)** pipeline running fully **locally**:

- **Embeddings:** [`mxbai-embed-large`](https://ollama.com/library/mxbai-embed-large) via Ollama
- **LLM:** [`gemma3:4b`](https://ollama.com/library/gemma3) (changeable)
- **Vector Store:** [Chroma](https://www.trychroma.com/) (persistent local DB)

You can ingest PDFs into a vector database and then query them with an LLM, all offline.

---

## 🚀 Features
- Load and split PDFs into chunks
- Generate embeddings locally with Ollama
- Persist vectors in a local Chroma DB
- Query using `gemma3:4b` (or any Ollama model)
- Sources returned with page references
- Works entirely offline (no external APIs)

---

## 📂 Project Structure
local-rag-ollama/
│── data/
│ └── attentionpaper.pdf # sample paper to test
│── ingest.py # script to build embeddings + DB
│── query.py # script to query the PDF with Ollama
│── requirements.txt # Python dependencies
│── .gitignore # ignores venv, DB, cache

---
## 2. Create & activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Windows PowerShell

## 3. Install dependencies
pip install -r requirements.txt

## 4. Install Ollama & models
a) Download Ollama
b) Pull required models:
ollama pull mxbai-embed-large
ollama pull gemma3:4b

Make sure Ollama server is running:
ollama serve

## 📥 Ingest PDF
Index the sample PDF (or add your own to data/):
python ingest.py
You should see logs like:
📄 Loading PDF…
✂️  Splitting into chunks…
🔢 Embedding with Ollama: mxbai-embed-large …
🧠 Building Chroma index…
✅ Ingestion complete.

## 🔎 Query

Ask questions against the PDF:
python query.py

❓ Query: Summarize the core idea of the Transformer in one sentence

🟢 Answer:
The Transformer is the first model to rely entirely on self-attention, removing recurrence and convolution.

📚 Sources:
- data/attentionpaper.pdf (p. 2)
- data/attentionpaper.pdf (p. 3)

## 🧩 Customize
Change LLM in query.py:
LLM_MODEL = "gemma3:4b"   # try "gemma2:2b", "qwen2.5:3b-instruct", "phi3.5-mini"
Tune retrieval in query.py:
K_NEIGHBORS = 6        # fewer chunks = faster
USE_MMR = True         # MMR gives diverse chunks
Swap PDF → add your own under data/ and re-run ingest.py.

## ⚡ Tips for Speed
1) Use smaller models (e.g., gemma2:2b) if CPU-only.
2) Reduce context window (NUM_CTX) and answer length (NUM_PREDICT).
3) Run Ollama with GPU acceleration (NVIDIA / Apple Silicon).

## 📜 License

MIT License — free to use, modify, and share.

## 🙌 Acknowledgements

Ollama
 for local LLMs and embeddings

LangChain
 for chaining components

Chroma
 for vector database





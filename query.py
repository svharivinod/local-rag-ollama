from pathlib import Path
from chromadb import PersistentClient
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# ---- paths & settings ----
ROOT = Path(__file__).parent.resolve()
PERSIST_DIR = str((ROOT / "chroma_db").resolve())
COLLECTION = "rag"
SETTINGS = Settings(anonymized_telemetry=False)

# ---- Ollama server ----
BASE_URL = "http://127.0.0.1:11434"   # change the port if you moved it

# ---- retrieval + perf knobs ----
USE_MMR = True          # True = more diverse results, usually better with fewer chunks
K_NEIGHBORS = 6         # keep small; sending 4–6 chunks is much faster than 10–12
MAX_CONTEXT_CHARS = 3500  # hard cap on context length (trim beyond this)

# ---- model perf knobs ----
LLM_MODEL = "gemma3:4b"   # try "gemma2:2b" or "qwen2.5:3b-instruct" for faster CPU
NUM_CTX = 1024            # smaller context => faster prefill; must be >= your context size
NUM_PREDICT = 128         # cap answer length
TEMPERATURE = 0.4

SYSTEM_PROMPT = """
You answer questions about the provided PDF.

Use the Context as your main evidence. If the exact answer is not stated,
infer the most likely answer from clues in the Context (titles, abstract,
headings, captions). Prefer precision over hedging.

Rules:
- Start with a direct answer in 1–2 sentences.
- If you inferred the answer, prefix with "Best-guess:".
- Add a short WHY with key phrase(s) and page numbers (p. N).
- If nothing is even weakly relevant, say "No relevant context found."
- Never invent page numbers.
Format:
Answer: <your answer>
Why: <brief evidence + pages>
Sources: p.<n>[, p.<n>...]
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Question: {input}\n\nContext:\n{context}")
])

def format_chunk(d):
    page = (d.metadata.get("page") or 0) + 1
    return f"(p. {page}) {d.page_content}"

def build_context(docs, max_chars=MAX_CONTEXT_CHARS):
    parts, used = [], 0
    for d in docs:
        chunk = format_chunk(d)
        if used + len(chunk) > max_chars:
            break
        parts.append(chunk)
        used += len(chunk)
    return "\n\n".join(parts)

def main():
    print("📁 Project root:", ROOT)
    print("💾 DB path:", PERSIST_DIR)

    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest", base_url=BASE_URL)

    client = PersistentClient(path=PERSIST_DIR, settings=SETTINGS)
    vectordb = Chroma(client=client, collection_name=COLLECTION, embedding_function=embeddings)

    cnt = vectordb._collection.count()
    print(f"🧪 Attached to collection '{COLLECTION}' — count: {cnt}")
    if cnt == 0:
        print("❌ Collection is empty. Run ingest.py first.")
        return

    # Faster LLM settings + streaming so you see tokens immediately
    llm = Ollama(
    model=LLM_MODEL,          # e.g., "gemma3:4b"
    base_url=BASE_URL,        # e.g., "http://127.0.0.1:11434"
    temperature=TEMPERATURE,  # 0.4
    num_ctx=NUM_CTX,          # 1024
    num_predict=NUM_PREDICT,  # 128
    )
    chain = prompt | llm | StrOutputParser()

    # Retrieval strategy
    if USE_MMR:
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": K_NEIGHBORS, "fetch_k": max(20, K_NEIGHBORS * 4), "lambda_mult": 0.5},
        )

    print("RAG ready. Type your questions (or 'exit').")
    while True:
        q = input("\n❓ Query: ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break

        if USE_MMR:
            docs = retriever.invoke(q)
            print(f"🔎 Retrieved {len(docs)} chunks via MMR")
        else:
            results = vectordb.similarity_search_with_score(q, k=K_NEIGHBORS)
            docs = [doc for doc, _ in results]
            print(f"🔎 Retrieved {len(docs)} chunks")

        if not docs:
            print("No docs retrieved. Try keywords from the title/abstract.")
            continue

        context = build_context(docs, MAX_CONTEXT_CHARS)

        print("🧮 Calling LLM…", flush=True)
        answer = chain.invoke({"input": q, "context": context})   # returns the full string now
        print("\n🟢 Answer:\n" + (answer.strip() or "[no text returned]"))  # <-- add this
        print("\n✅ Done.")

        # Optional: list distinct sources once
        seen = set()
        print("\n📚 Sources:")
        for d in docs:
            src = d.metadata.get("source"), (d.metadata.get("page") or 0) + 1
            if src in seen: 
                continue
            seen.add(src)
            print(f"- {src[0]} (p. {src[1]})")

if __name__ == "__main__":
    main()

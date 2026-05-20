import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ── Module-level cache ──────────────────────────────────────────────────
_embeddings_cache = None
_vectorstore_cache = None

# Determine which embeddings to use
def get_embeddings():
    global _embeddings_cache
    if _embeddings_cache is not None:
        return _embeddings_cache

    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if groq_key:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings_cache = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    elif gemini_key:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        _embeddings_cache = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=gemini_key,
        )
    elif openai_key:
        from langchain_openai import OpenAIEmbeddings
        _embeddings_cache = OpenAIEmbeddings()
    else:
        raise ValueError(
            "No embedding API key found. Set GROQ_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY in your .env file."
        )
    return _embeddings_cache

FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index")
PROVIDER_MARKER = os.path.join(FAISS_INDEX_PATH, ".provider")


def _current_provider() -> str:
    """Return a string identifying the current embedding provider."""
    if os.getenv("GROQ_API_KEY"):
        return "huggingface"
    elif os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return "google"
    elif os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "unknown"


def build_faiss_index(force: bool = False):
    provider = _current_provider()

    # Check if index exists and was built with the same provider
    if os.path.exists(FAISS_INDEX_PATH) and not force:
        if os.path.exists(PROVIDER_MARKER):
            with open(PROVIDER_MARKER, "r") as f:
                saved_provider = f.read().strip()
            if saved_provider == provider:
                return  # Index exists and matches current provider
            else:
                print(f"Embedding provider changed ({saved_provider} -> {provider}). Rebuilding FAISS index...")
        else:
            # No marker = old index, rebuild to be safe
            print("No provider marker found. Rebuilding FAISS index...")

    print("Building FAISS index...")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))

    docs = []
    for file in pdf_files:
        loader = PyPDFLoader(file)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)

    # Save provider marker
    with open(PROVIDER_MARKER, "w") as f:
        f.write(provider)

    print(f"FAISS index built successfully with {len(splits)} chunks ({provider} embeddings).")

def retrieve_policy_context(query: str, k: int = 3) -> str:
    global _vectorstore_cache
    build_faiss_index()
    
    if _vectorstore_cache is None:
        embeddings = get_embeddings()
        _vectorstore_cache = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("FAISS vectorstore loaded and cached.")
    
    docs = _vectorstore_cache.similarity_search(query, k=k)
    
    context_parts = []
    for i, doc in enumerate(docs):
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", 0) + 1
        content = doc.page_content.replace("\n", " ")
        context_parts.append(f"[Source: {source}, Page: {page}]\n{content}")
        
    return "\n\n".join(context_parts)


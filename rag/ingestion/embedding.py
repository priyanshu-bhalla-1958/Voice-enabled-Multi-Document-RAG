from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pathlib import Path

DB_NAME = str(Path(__file__).parent.parent / "vector_db")
COLLECTION_NAME = "docs"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def normalize_docs(docs):
    normalized = []

    for d in docs:
        if isinstance(d, dict):
            normalized.append(
                Document(
                    page_content=d.get("text", ""),
                    metadata=d.get("metadata", {})
                )
            )
        else:
            normalized.append(d)

    return normalized


def create_embeddings(chunks):
    normalizedChunks = normalize_docs(chunks)

    vectorstore = Chroma(
        persist_directory=DB_NAME,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    vectorstore.add_documents(normalizedChunks)
    
    count = vectorstore._collection.count()
    print(f"✅ Stored {count} chunks in vector DB")

    return vectorstore
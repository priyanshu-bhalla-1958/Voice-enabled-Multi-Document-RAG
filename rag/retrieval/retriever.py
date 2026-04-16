import json
from pathlib import Path
from openai import OpenAI
from config import OPENAI_API_KEY
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


DB_NAME = str(Path(__file__).parent.parent / "vector_db")
client = OpenAI(api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
RETRIEVAL_K = 20
FINAL_K = 10
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings, collection_name="docs")
retriever = vectorstore.as_retriever()

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

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant who answers questions accurately and concisely.
You are chatting with a user about the processed document uploaded.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the processed document that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""

def rewrite_query(question, history=[]):
    """Rewrite the user's question to be a more specific question that is more likely to surface relevant content in processed documents."""
    message = f"""
You are in a conversation with a user, answering questions about the processed documents which user has uploaded.
You are about to look up information in a processed document to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a short, refined question that you will use to search the processed documents.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
IMPORTANT: Respond ONLY with the precise document query, nothing else.
"""
    response = client.chat.completions.create(model="gpt-4.1-nano", messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content

def rerank(query, docs):
    scored = []
    docs = normalize_docs(docs)

    for doc in docs:
        prompt = f"""
        You are an expert retrieval ranking system. Your job is to evaluate how relevant a document chunk is to a user query.
        The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
        You must rank order the provided chunks by relevance to the question, and assign a relevance score to each chunk from 0 to 10 based on the following criteria:

        GOAL:
        Given a QUERY and a DOCUMENT CHUNK, assign a relevance score from 0 to 10.

        SCORING RULES:
        - 10 = Directly answers the query with precise, useful information.
        - 8 to 9 = Highly relevant, but may miss minor details.
        - 6 to 7 = Partially relevant, contains useful but incomplete information.
        - 3 to 5 = Weak relevance, loosely related but not useful for answering.
        - 1 to 2 = Barely related.
        - 0 = Completely irrelevant.

        IMPORTANT GUIDELINES:
        - Prioritize factual alignment with the query.
        - Prefer chunks that contain specific answers over general discussion.
        - Penalize vague, generic, or repetitive content.
        - Penalize chunks that do not directly help answer the query.
        - Do NOT hallucinate missing information.
        - Do NOT assume context outside the given chunk.
        - Be strict in scoring.

        OUTPUT FORMAT (STRICT):
        Return ONLY a valid JSON object:
        {{
            "score": <integer between 0 and 10>,
            "reason": "<short explanation (1 sentence)>"    
        }}

        Query: {query}
        Document: {doc.page_content}
        """

        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(content)
            score = int(parsed["score"])
        except Exception as e:
            print("⚠️ JSON parsing failed:", content, e)
            score = 0 

        scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored[:5]]

def merge_chunks(chunks, reranked):
    normalizedChunks = normalize_docs(chunks)
    normalizedReRanked = normalize_docs(reranked)
    merged = normalizedChunks[:]
    existing = [chunk.page_content for chunk in normalizedChunks]
    for chunk in normalizedReRanked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged

def fetch_context_unranked(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question, k=RETRIEVAL_K)

def fetch_context(original_question):
    rewritten_question = rewrite_query(original_question)
    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)
    chunks = merge_chunks(chunks1, chunks2)
    reranked = rerank(original_question, chunks)
    return reranked[:FINAL_K]

def make_rag_messages(question, history, chunks):
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )

def answer_question(question: str, history: list[dict] = []) -> tuple[str, list]:
    """
    Answer a question using RAG and return the answer and the retrieved context
    """
    chunks = fetch_context(question)
    messages = make_rag_messages(question, history, chunks)
    response = client.chat.completions.create(model="gpt-4.1-nano", messages=messages)
    return response.choices[0].message.content, chunks
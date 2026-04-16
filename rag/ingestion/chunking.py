from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from config import OPENAI_API_KEY
from langchain_core.documents import Document

client = OpenAI(api_key=OPENAI_API_KEY)

def semantic_chunk(document):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )
    return splitter.split_text(document.page_content)


def llm_restructure(chunk):
    prompt = f"""
    Improve this chunk for retrieval:
    - Add a brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query
    - Add a summary, a few sentences summarizing the content of this chunk to answer common questions
    - Keep the original text of this chunk from the provided document, exactly as is, not changed in any way

    Text:
    {chunk}
    """

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}]
    )
    print("LLM Restructure response:")
    return response.choices[0].message.content


def create_chunks(documents):
    final_chunks = []

    for doc in documents:
        chunks = semantic_chunk(doc)
        print(f"Document: split into {len(chunks)} chunks.")

        for chunk in chunks:
            structured = llm_restructure(chunk)
            final_chunks.append(
                Document(
                    page_content=structured,
                    metadata=doc.metadata
                )
            )

    return final_chunks
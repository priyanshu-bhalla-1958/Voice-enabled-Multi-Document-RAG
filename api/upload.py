from langchain_core.documents import Document
from fastapi import FastAPI, UploadFile, File
import os
from cryptography.fernet import Fernet
from rag.ingestion.chunking import create_chunks
from rag.ingestion.embedding import create_embeddings
from rag.ingestion.processor import process_document
from storage.security import load_key

app = FastAPI()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

UPLOAD_DIR = os.path.join(BASE_DIR, "storage/uploads")
ENCRYPTED_DIR = os.path.join(BASE_DIR, "storage/encrypted")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ENCRYPTED_DIR, exist_ok=True)


def clear_storage():
    folders = [
        "storage/encrypted",
        "storage/decrypted",
        "storage/processed"
    ]

    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)

            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Step 1: Clear old data (optional but recommended)
        clear_storage()

        filename = os.path.basename(file.filename)
        contents = await file.read()

        # Step 2: Encrypt
        key = load_key()
        fernet = Fernet(key)
        encrypted_data = fernet.encrypt(contents)

        encrypted_path = os.path.join(ENCRYPTED_DIR, filename + ".enc")

        with open(encrypted_path, "wb") as f:
            f.write(encrypted_data)

        # 🔥 Step 3: Process document → markdown
        result = process_document(encrypted_path)
        md_path = result["markdown_path"]

        # 🔥 Step 4: Load markdown as document
        documents = [
            Document(
                page_content=open(md_path, "r", encoding="utf-8").read(),
                metadata={"source": md_path}
            )
        ]

        # 🔥 Step 5: Chunking
        chunks = create_chunks(documents)
        print(f"Total chunks created: {len(chunks)}")

        # 🔥 Step 6: Embedding + store in Chroma
        create_embeddings(chunks)
        print("Embeddings created and stored in Chroma.")

        return {
            "status": "success",
            "filename": filename,
            "markdown_path": md_path,
            "chunks_created": len(chunks)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
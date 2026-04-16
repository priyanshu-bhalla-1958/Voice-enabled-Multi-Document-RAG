import os
import base64
import mimetypes
import fitz
from cryptography.fernet import Fernet
from storage.security import load_key
from unstructured.partition.auto import partition
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)



# Base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DECRYPTED_DIR = os.path.join(BASE_DIR, "storage/decrypted")
PROCESSED_DIR = os.path.join(BASE_DIR, "storage/processed")

os.makedirs(DECRYPTED_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ========================
# 1. Decrypt File
# ========================
def decrypt_file(encrypted_path):
    key = load_key()
    fernet = Fernet(key)

    filename = os.path.basename(encrypted_path).replace(".enc", "")
    decrypted_path = os.path.join(DECRYPTED_DIR, filename)

    with open(encrypted_path, "rb") as f:
        encrypted_data = f.read()

    decrypted_data = fernet.decrypt(encrypted_data)

    with open(decrypted_path, "wb") as f:
        f.write(decrypted_data)

    return decrypted_path


# ========================
# 2.0 Extract Text
# ========================

def extract_text(file_path):
    elements = partition(filename=file_path)

    text_chunks = []
    for el in elements:
        if hasattr(el, "text") and el.text:
            text_chunks.append(el.text)

    return "\n".join(text_chunks)

# ========================
# 2.0 Extract Text from Images
# ========================
system_prompt = "You are a helpful assistant that extracts text from images. Preserve the structure and formatting as much as possible. Do not hallucinate or add any information that is not present in the image."

def extract_text_from_image(image_path):
    print(f"Extracting text from image: {image_path} using OpenAI OCR...")
    try:
        print("Loading image and encoding to base64...")
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        # Encode to base64
        base64_image = base64.b64encode(image_bytes).decode()

        # Detect MIME type dynamically
        mime_type, _ = mimetypes.guess_type(image_path)

        if mime_type is None:
            raise ValueError("Unsupported image type")

        image_url = f"data:{mime_type};base64,{base64_image}"

        response = client.responses.create(
            model="gpt-4.1-nano",
            input=[
                {"role": "system", "content": system_prompt},
                {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Extract all text from this image. Preserve structure. Do not hallucinate."
                    },
                    {
                        "type": "input_image",
                        "image_url": image_url
                    }
                ],
            }]
        )
        print(f"Received response from OpenAI OCR: {response}")
        text = response.output_text
        print(f"Extracted Text:\n{text}")
        return text

    except Exception as e:
        print("OpenAI OCR failed:", e)
        return ""


# ========================
# 2.1 Extract Text from Images
# ========================

def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    image_paths = []

    for i, page in enumerate(doc):
        print(i, type(page), page)
        pix = page.get_pixmap(dpi=300)  # high quality
        img_path = f"temp_page_{i}.png"
        pix.save(img_path)
        image_paths.append(img_path)

    return image_paths

def extract_text_from_pdf(pdf_path):
    image_paths = pdf_to_images(pdf_path)
    full_text = []

    for img_path in image_paths:
        text = extract_text_from_image(img_path)  # your OpenAI OCR
        full_text.append(text)
        os.remove(img_path)

    return "\n".join(full_text)
# ========================
# 2.2 Smart Extract
# ========================

def smart_extract(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".pdf"]:
        return extract_text_from_pdf(file_path)

    elif ext in [".jpg", ".png", ".jpeg"]:
        return extract_text_from_image(file_path)

    elif ext in [".docx", ".txt"]:
        return extract_text(file_path)

    else:
        return "Unsupported file type"

# ========================
# 3. Convert to Markdown
# ========================
def convert_to_markdown(text):
    lines = text.split("\n")

    md_lines = ["# Document\n"]

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Heuristic: short uppercase lines → headings
        if len(line) < 60 and line.isupper():
            md_lines.append(f"\n## {line}\n")
        else:
            md_lines.append(line)

    return "\n".join(md_lines)


# ========================
# 4. Save Markdown
# ========================
def save_markdown(md_text, original_filename):
    filename = os.path.basename(original_filename).replace(".enc", "")
    filename = filename.split(".")[0]  # remove extension

    md_path = os.path.join(PROCESSED_DIR, filename + ".md")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    return md_path


# ========================
# 5. Full Pipeline
# ========================
def process_document(encrypted_path):
    try:
        # Step 1: Decrypt
        decrypted_path = decrypt_file(encrypted_path)

        # Step 2: Extract smart text
        text = smart_extract(decrypted_path)

        # Step 3: Convert to markdown
        md_text = convert_to_markdown(text)

        # Step 4: Save markdown
        md_path = save_markdown(md_text, encrypted_path)

        # Cleanup decrypted file (important 🔥)
        os.remove(decrypted_path)

        return {
            "status": "success",
            "markdown_path": md_path
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
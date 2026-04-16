import gradio as gr
import requests
from constants import API_URL
from rag.retrieval.retriever import answer_question

theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])


def upload_to_api(file):
    if file is None:
        return "⚠️ Please select a file first."

    try:
        import os

        file_path = file if isinstance(file, str) else file.name
        filename = os.path.basename(file_path)

        with open(file_path, "rb") as f:
            response = requests.post(
                API_URL,
                files={"file": (filename, f)},
                timeout=120,
            )

        result = response.json()
        print(f"API Response: {result}")

        if result["status"] == "success":
            processed_path = result.get("markdown_path", "N/A")

            return f"""
        ✅ Uploaded: {result['filename']}
        📄 Markdown generated: {processed_path}
"""
        else:
            return f"❌ Error: {result['message']}"

    except requests.exceptions.ConnectionError:
        return "❌ Cannot reach the server. Is the API running?"

    except Exception as e:
        return f"❌ Upload failed: {e}"


def chat(user_message, history):
    if not user_message.strip():
        return "", history

    openai_history = []
    for msg in history:
        openai_history.append({"role": msg["role"], "content": msg["content"]})

    answer, _ = answer_question(user_message, openai_history)

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": answer})
    return "", history


with gr.Blocks(theme=theme, title="Document RAG Assistant") as app:
    gr.Markdown(
        "#Document RAG Assistant\n"
        "Upload a document on the left, then ask questions about it on the right."
    )

    with gr.Row(equal_height=True):
        # ── Left column: Upload ──────────────────────────────
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### Upload Document")
            file_input = gr.File(
                label="File / Image Upload",
                file_types=[".pdf", ".docx", ".txt"],
            )
            upload_btn = gr.Button("Upload & Process", variant="primary")
            upload_status = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2,
            )

            upload_btn.click(
                upload_to_api,
                inputs=file_input,
                outputs=upload_status,
            )

        # ── Right column: Chatbot ────────────────────────────
        with gr.Column(scale=2, min_width=480):
            gr.Markdown("### Chat with your Document")
            chatbot = gr.Chatbot(
                height=480,
                placeholder="Upload a document and start asking questions…",
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Type your question here…",
                    show_label=False,
                    scale=9,
                    container=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            send_btn.click(
                chat,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot],
            )
            msg_input.submit(
                chat,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot],
            )

app.launch()
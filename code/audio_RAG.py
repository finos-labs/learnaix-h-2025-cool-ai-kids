import gradio as gr
import os
import logging

from perform_recording import (
    speech_recognition_record_audio,
    speech_recognition_transcribe_audio
)

from vector_rag_implementation import (
    read_pdf, 
    chunk_text, 
    create_faiss_index, 
    query_rag
)

from text_to_audio import pyttsx3_text_to_speech

# ------------------------
# Setup logging
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("recorder.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def transcribe_and_find_ans():
    """Directly transcribe audio using speech recognition and query RAG."""
    logging.info("Stop button clicked. Attempting to transcribe...")
    status, asked_query = speech_recognition_transcribe_audio()
    logging.info(f"Transcription result: {asked_query}")

    query_resp = query_rag(asked_query)
    # query_resp = asked_query          # just for testing
    return status, asked_query, query_resp


# ------------------------
# Process document
# ------------------------
def process_document(file_path):
    """Read a PDF, chunk it, and create FAISS index."""
    if file_path:
        logging.info(f"Processing document: {file_path}")
        text = read_pdf(file_path)
        chunks = chunk_text(text)
        create_faiss_index(chunks)
        logging.info("RAG system ready.")
        return f"✅ Processing done for: {os.path.basename(file_path)}"
    return "⚠️ No file to process."


# ------------------------
# Post-process RAG response (TTS)
# ------------------------
def post_process_response(query_response_box: str):
    """Convert the response text into speech."""
    if query_response_box:
        pyttsx3_text_to_speech(query_response_box)
    return "Go on! Ask the next question!!"


# ------------------------
# Gradio UI
# ------------------------
with gr.Blocks(theme=gr.themes.Soft(primary_hue="purple", secondary_hue="pink")) as demo:
    with gr.Column(elem_classes="container"):
        gr.Markdown(
            """
            <div style="text-align: center; font-size:14px; line-height:1.3em; margin:0; padding:0;">
                <h2 style="color:#7e3ff2; margin-bottom:6px;">🎤 Audio RAG based Q&A.</h2>
                <p style="margin:4px 0;"><b>1. Upload PDF</b> → 📄 Upload + 📄 Process Document</p>
                <p style="margin:4px 0;"><b>2. Ask</b> → 🤚 Ask Question, then ֎ Generate Response</p>
                <p style="margin:4px 0;"><b>3. Answer</b> → See text + hear speech</p>
                <p style="margin:4px 0;"><b>4. Repeat</b> → Keep asking without re-uploading</p>
            </div>
            """,
            elem_classes="title"
        )

        # PDF upload
        pdf_file = gr.File(label="📄 Upload PDF", file_types=[".pdf"], type="filepath")

        # Process document button
        process_btn = gr.Button("📄 Process Document", interactive=False)
        process_status = gr.Label(label="Process Status", elem_classes="status-box")

        # Recording buttons
        with gr.Row():
            start_btn = gr.Button("🤚 Ask Question", elem_classes="start-btn")
            stop_btn = gr.Button("֎ Generate Response", elem_classes="stop-btn")

        status = gr.Label(label="Status", elem_classes="status-box")

        # Query + Response boxes
        asked_query_box = gr.Textbox(
            label="🧐 Your Query",
            interactive=False,
            lines=5,
            max_lines=20,
            elem_classes="dynamic-box"
        )
        query_response_box = gr.Textbox(
            label="🗣️ Response to the Query",
            interactive=False,
            lines=8,
            max_lines=30,
            elem_classes="dynamic-box"
        )

        # Enable "Process document" button only when a PDF is uploaded
        def enable_button(file):
            return gr.update(interactive=bool(file))

        pdf_file.upload(enable_button, inputs=pdf_file, outputs=process_btn)

        # Button events
        process_btn.click(process_document, inputs=pdf_file, outputs=process_status)
        start_btn.click(speech_recognition_record_audio, outputs=status)
        stop_btn.click(transcribe_and_find_ans, outputs=[status, asked_query_box, query_response_box])

        # Auto TTS after response
        query_response_box.change(
            post_process_response,
            inputs=query_response_box,
            outputs=process_status
        )


# ------------------------
# Custom CSS
# ------------------------
demo.css = """
.container {
    max-width: 700px;
    margin: auto;
    padding: 20px;
}

.start-btn {
    background-color: #34d399 !important;
    color: white !important;
    font-size: 18px;
    border-radius: 8px;
}

.stop-btn {
    background-color: #ef4444 !important;
    color: white !important;
    font-size: 18px;
    border-radius: 8px;
}

.status-box {
    font-size: 11px !important;
    padding: 6px;
    color: #4b5563;
}

.dynamic-box textarea {
    font-size: 14px !important;
    line-height: 1.5em;
    resize: vertical !important;
    overflow-y: auto !important;
}
"""

demo.launch()

import os
import io
import json
import atexit
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from openai import OpenAI
import streamlit as st
from docx import Document

# Load env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
MAIN_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

client = OpenAI(api_key=OPENAI_API_KEY)

# Write service account to temp file
SERVICE_ACCOUNT_FILE = "service_account.json"
with open(SERVICE_ACCOUNT_FILE, "w") as f:
    f.write(SERVICE_ACCOUNT_JSON)

@st.cache_resource
def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    return build('drive', 'v3', credentials=creds)

def list_subfolders(service, parent_id):
    results = service.files().list(
        q=f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder'",
        fields="files(id, name)").execute()
    return results.get('files', [])

def list_pdfs_in_folder(service, folder_id):
    results = service.files().list(
        q=f"'{folder_id}' in parents and mimeType='application/pdf'",
        fields="files(id, name)").execute()
    return results.get('files', [])

def download_pdf_as_bytes(service, file_id):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh.read()

def extract_text_from_pdf_bytes(pdf_data):
    reader = PdfReader(io.BytesIO(pdf_data))
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def create_docx_from_text(text):
    doc = Document()
    for para in text.split("\n"):
        doc.add_paragraph(para)
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf

# UI
st.title("Audit Assistant with OpenAI")
service = get_drive_service()

# Split layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("PDF Selection")

    # Step 1: Subfolder selection
    subfolders = list_subfolders(service, MAIN_FOLDER_ID)
    folder_names = [f['name'] for f in subfolders]
    selected_folder_names = st.multiselect("Select subfolder(s)", folder_names)

    # Step 2: List PDFs from Drive
    pdfs = []
    for name in selected_folder_names:
        folder_id = next(f['id'] for f in subfolders if f['name'] == name)
        pdfs_in_folder = list_pdfs_in_folder(service, folder_id)
        for pdf in pdfs_in_folder:
            pdf['folder_name'] = name
            pdfs.append(pdf)

    # Step 3: Upload local PDFs
    uploaded_files = st.file_uploader("Or upload local PDF files", type=["pdf"], accept_multiple_files=True)

    # Step 4: Model choice
    model_choice = st.radio("Choose OpenAI Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4o"], horizontal=True)
    if model_choice in ["gpt-4", "gpt-4o"]:
        st.warning("⚠️ Using this model will consume more tokens and may increase costs.")

    # Step 5: User query input (always visible)
    user_input = st.text_input("What would you like to ask or do?")

with col2:
    # Step 6: Process and show
    combined_text = ""

    for pdf in pdfs:
        pdf_data = download_pdf_as_bytes(service, pdf['id'])
        combined_text += f"\n\n--- {pdf['folder_name']} / {pdf['name']} ---\n"
        combined_text += extract_text_from_pdf_bytes(pdf_data)

    for uploaded_file in uploaded_files:
        combined_text += f"\n\n--- uploaded / {uploaded_file.name} ---\n"
        combined_text += extract_text_from_pdf_bytes(uploaded_file.read())

    if combined_text:
        st.subheader("PDF Content Preview")
        st.text_area("Extracted Text", value=combined_text[:2000], height=300)

    if user_input and combined_text:
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": "You're an expert in auditing, compliance, and document comparison. Respond precisely to the user's query."},
                    {"role": "user", "content": f"The following is content from selected PDF(s):\n{combined_text[:10000]}\n\nUser instruction:\n{user_input}"}
                ]
            )
            final_response = response.choices[0].message.content
            st.subheader("OpenAI's Response")
            st.write(final_response)

            docx_buffer = create_docx_from_text(final_response)
            st.download_button("Download as DOCX", data=docx_buffer, file_name="openai_response.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# Cleanup temp file
atexit.register(lambda: os.remove(SERVICE_ACCOUNT_FILE) if os.path.exists(SERVICE_ACCOUNT_FILE) else None)

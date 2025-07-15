import streamlit as st
from PyPDF2 import PdfReader
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import json
from dotenv import load_dotenv
import ssl
from openai import OpenAI

# --- Load environment variables ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID") or 'your-folder-id'

# --- Fix SSL issues (macOS/legacy) ---
ssl._create_default_https_context = ssl._create_unverified_context

# --- Initialize OpenAI client ---
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Write service account JSON to temp file ---
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

def list_pdfs(service):
    results = service.files().list(
        q=f"'{FOLDER_ID}' in parents and mimeType='application/pdf'",
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

# --- UI ---
st.title("Audit Assistant with OpenAI")

service = get_drive_service()
pdfs = list_pdfs(service)

if not pdfs:
    st.warning("No PDFs found in the folder.")
else:
    file_names = [file['name'] for file in pdfs]
    selected_file_names = st.multiselect("Select PDF(s)", file_names)

    if selected_file_names:
        combined_text = ""
        for name in selected_file_names:
            file_id = next(file['id'] for file in pdfs if file['name'] == name)
            pdf_data = download_pdf_as_bytes(service, file_id)
            combined_text += f"\n\n--- {name} ---\n"
            combined_text += extract_text_from_pdf_bytes(pdf_data)

        st.subheader("PDF Content Preview")
        st.text_area("Extracted Text", value=combined_text[:2000], height=300)

        user_input = st.text_input("What would you like to ask or do?")

        if user_input:
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",  # use gpt-4 if needed
                    messages=[
                        {"role": "system", "content": "You're an expert in auditing, compliance, and document comparison. Respond precisely to the user's query."},
                        {"role": "user", "content": f"The following is content from selected PDF(s):\n{combined_text[:10000]}\n\nUser instruction:\n{user_input}"}
                    ]
                )
                st.subheader("OpenAI's Response")
                st.write(response.choices[0].message.content)

# Optional cleanup
import atexit
atexit.register(lambda: os.remove(SERVICE_ACCOUNT_FILE) if os.path.exists(SERVICE_ACCOUNT_FILE) else None)

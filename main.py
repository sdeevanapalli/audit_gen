import os
import io
import json
import atexit
import tempfile
from dotenv import load_dotenv
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

if not OPENAI_API_KEY:
    st.error("❌ OpenAI API key not found in environment.")
    st.stop()

if not SERVICE_ACCOUNT_JSON or not MAIN_FOLDER_ID:
    st.error("❌ Missing Google Drive credentials or folder ID.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

try:
    parsed_json = json.loads(SERVICE_ACCOUNT_JSON)
    parsed_json["private_key"] = parsed_json["private_key"].replace("\\n", "\n")
    temp_service_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False)
    json.dump(parsed_json, temp_service_file)
    temp_service_file.close()
    SERVICE_ACCOUNT_FILE = temp_service_file.name
except Exception as e:
    st.error(f"❌ Failed to process service account key: {e}")
    st.stop()

atexit.register(lambda: os.remove(SERVICE_ACCOUNT_FILE) if os.path.exists(SERVICE_ACCOUNT_FILE) else None)

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

def list_txt_in_folder(service, folder_id):
    results = service.files().list(
        q=f"'{folder_id}' in parents and mimeType='text/plain'",
        fields="files(id, name)").execute()
    return results.get('files', [])

def download_txt_as_text(service, file_id):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh.read().decode('utf-8')

def create_docx_from_text(text):
    doc = Document()
    for para in text.split("\n"):
        doc.add_paragraph(para)
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf

def chunk_text(text, max_chars=3000):
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
        current_chunk += para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# UI Layout
col_1, col_logo, col_2 = st.columns([1, 6, 1])
with col_logo:
    st.image("Masthead.png", width=500)

st.title("Internal Audit Officer (IAO)")

service = get_drive_service()

st.subheader("Reference Document Selection")

subfolders = list_subfolders(service, MAIN_FOLDER_ID)
name_to_id = {f['name']: f['id'] for f in subfolders}
selected_folder_names = st.multiselect("Select Set of Documents", list(name_to_id.keys()))

txts = []
if selected_folder_names:
    for name in selected_folder_names:
        folder_id = name_to_id[name]
        txts_in_folder = list_txt_in_folder(service, folder_id)
        for txt in txts_in_folder:
            txt['folder_name'] = name
            txts.append(txt)

uploaded_files = st.file_uploader("Upload Proposal Document", type=["txt"], accept_multiple_files=True)

# Preset prompts (stored in hidden session_state)
if "preset_prompt" not in st.session_state:
    st.session_state["preset_prompt"] = ""

preset_queries = {
    "Finance Concurrence": "Please examine the uploaded document according to govt guidelines and GFR, and give finance concurrence.",
    "Payment Proposal": "Please create a payment proposal for the uploaded document according to the template named payment proposal.pdf, in line with the guidelines document and GFR.",
    "Internal Audit": "Please draft an internal audit document for the uploaded proposal according to the internal audit manual and other guidelines."
}

def process_query(query_to_send):
    if not query_to_send:
        st.warning("⚠️ Please enter a query or select a preset prompt.")
        return
    if not combined_text:
        st.warning("⚠️ Please select or upload at least one TXT file to proceed.")
        return

    st.subheader("Response")
    with st.spinner("Thinking..."):
        try:
            chunks = chunk_text(combined_text, max_chars=3000)
            # ✅ Combine all chunks into one long prompt
            full_combined_text = ""
            for i, chunk in enumerate(chunks):
                full_combined_text += f"\n\n[Content Chunk {i+1}]\n{chunk}"

            # ✅ Send all at once as a single unified request
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert in auditing, compliance, and document comparison. Provide a single, consolidated response to the user’s instruction after analyzing the full content."},
                    {"role": "user", "content": f"{full_combined_text}\n\nUser instruction:\n{query_to_send}"}
                ]
            )

            final_response = response.choices[0].message.content
            st.write(final_response)

            # ✅ DOCX download
            docx_buffer = create_docx_from_text(final_response)
            st.download_button(
                "Download as DOCX",
                data=docx_buffer,
                file_name="openai_response.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        except Exception as e:
            st.error(f"❌ OpenAI request failed: {e}")

st.subheader("Quick Prompt")
col1, col2, col3 = st.columns(3)
quick_prompt_triggered = False
with col1:
    if st.button("Finance Concurrence"):
        st.session_state["preset_prompt"] = preset_queries["Finance Concurrence"]
        quick_prompt_triggered = True
with col2:
    if st.button("Payment Proposal"):
        st.session_state["preset_prompt"] = preset_queries["Payment Proposal"]
        quick_prompt_triggered = True
with col3:
    if st.button("Internal Audit"):
        st.session_state["preset_prompt"] = preset_queries["Internal Audit"]
        quick_prompt_triggered = True

# Prepare combined text from selected/uploaded TXT files
combined_text = ""
for txt in txts:
    txt_data = download_txt_as_text(service, txt['id'])
    combined_text += f"\n\n--- {txt['folder_name']} / {txt['name']} ---\n"
    combined_text += txt_data

for uploaded_file in uploaded_files:
    combined_text += f"\n\n--- uploaded / {uploaded_file.name} ---\n"
    combined_text += uploaded_file.read().decode('utf-8')

# Model selection
model_choice = st.radio("Select Model", ["alpha", "beta", "gamma"], horizontal=True)
if model_choice == 'alpha':
    model = "gpt-3.5-turbo"
elif model_choice == 'beta':
    model = 'gpt-4'
elif model_choice == 'gamma':
    model = 'gpt-4o'

if quick_prompt_triggered:
    process_query(st.session_state["preset_prompt"])

user_input = st.text_area(
    "Type your Query Here",
    height=140,
    placeholder="Type your query here..."
)

submitted = st.button("🔍 Submit (Ctrl/Cmd + Enter)")

if submitted:
    process_query(user_input.strip())

if not (submitted or quick_prompt_triggered):
    if combined_text:
        st.subheader("Content Preview (Before Search)")
        st.text_area("Extracted Text (max. 2000 char)", value=combined_text[:2000], height=300)
    else:
        st.info("No TXT content to preview yet.")

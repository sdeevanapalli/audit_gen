import os
import io
import tempfile
import atexit
import json
import streamlit as st

from openai import OpenAI
from docx import Document
from pdfminer.high_level import extract_text as extract_pdf_text
from dotenv import load_dotenv
import tiktoken

# GOOGLE DRIVE
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# --------- ENV CONFIG AND SETUP -------
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
SERVICE_ACCOUNT_JSON = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
DRIVE_MAIN_FOLDER_ID = os.getenv('GOOGLE_DRIVE_FOLDER_ID')

st.set_page_config(page_title="Internal Audit Officer", layout="wide")

MODEL_MAP = {
    "Alpha (provides a brief and concise summary; optimized for fast responses)": "gpt-3.5-turbo",
    "Beta (offers a balanced output combining brevity and detail)": "gpt-4",
    "Gamma (delivers comprehensive and detailed analysis; may take longer to respond)": "gpt-4o"
}

PRESET_QUERIES = {
    "Finance Concurrence": "Please examine the uploaded document according to govt guidelines and GFR uploaded in the google drive and selected here, and give finance concurrence accordingly.",
    "Payment Proposal": "Please create a payment proposal for the uploaded document according to the template named payment proposal.pdf, in line with the guidelines document and GFR which are uploaded on google drive.",
    "Internal Audit": "Please draft an internal audit document for the uploaded proposal according to the internal audit manual and other guidelines which are uploaded on the google drive accordingly."
}

CONTEXT_CHUNKS = 8
CHUNK_CHAR_LIMIT = 1000
PROPOSAL_CHAR_LIMIT = 2200
TOKEN_BUDGET = 8000
MAX_RESPONSE_TOKENS = 1024

client = OpenAI(api_key=API_KEY)

# ---------------- HELPERS --------------

def count_tokens(text, model="gpt-4o"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def get_drive_service():
    parsed_json = json.loads(SERVICE_ACCOUNT_JSON)
    parsed_json["private_key"] = parsed_json["private_key"].replace("\\n", "\n")
    temp_service_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False)
    json.dump(parsed_json, temp_service_file)
    temp_service_file.close()
    SERVICE_ACCOUNT_FILE = temp_service_file.name
    atexit.register(lambda: os.remove(SERVICE_ACCOUNT_FILE) if os.path.exists(SERVICE_ACCOUNT_FILE) else None)
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=['https://www.googleapis.com/auth/drive.readonly'],
    )
    return build('drive', 'v3', credentials=creds)

def list_subfolders(service, parent_id):
    results = service.files().list(
        q=f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields="files(id, name)").execute()
    return results.get('files', [])

def list_txt_in_folder(service, folder_id):
    results = service.files().list(
        q=f"'{folder_id}' in parents and mimeType='text/plain' and trashed=false",
        pageSize=500,
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
    return fh.read().decode('utf-8', errors='ignore')

def parse_uploaded_file(uploaded_file):
    fname = uploaded_file.name.lower()
    if fname.endswith('.txt'):
        try:
            return uploaded_file.read().decode('utf-8', errors='ignore')
        except Exception:
            return ""
    elif fname.endswith('.docx'):
        try:
            doc = Document(io.BytesIO(uploaded_file.read()))
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception:
            return ""
    elif fname.endswith('.pdf'):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp.close()
                text = extract_pdf_text(tmp.name)
                os.remove(tmp.name)
                return text
        except Exception:
            return ""
    else:
        return ""

# ----------- UI LAYOUT -----------------

colx1, colx2, colx3 = st.columns([1,2,1])
with colx2:
    st.image("Masthead.png", use_container_width=True)
    st.markdown("<h1 style='text-align: center;'>Internal Audit Officer (IAO)</h1>", unsafe_allow_html=True)

try:
    drive_service = get_drive_service()
except Exception as e:
    st.error(f"Google Drive authentication/setup failed: {e}")
    st.stop()

# -------- Subfolder Selection ------------
st.subheader("1. Select Reference Subfolders")
subfolders = list_subfolders(drive_service, DRIVE_MAIN_FOLDER_ID)
if not subfolders:
    st.warning("No subfolders found in project.")
    st.stop()
subfolder_names = [f['name'] for f in subfolders]
subfolder_map = {f['name']: f['id'] for f in subfolders}

selected_subfolders = st.multiselect("Choose one or more subfolders", subfolder_names)

if "reference_docs" not in st.session_state:
    st.session_state.reference_docs = []

if selected_subfolders:
    if st.button(f"Fetch and combine files from: {', '.join(selected_subfolders)}"):
        docs = []
        with st.spinner("Fetching reference files..."):
            for subfolder_name in selected_subfolders:
                files_in_sub = list_txt_in_folder(drive_service, subfolder_map[subfolder_name])
                for file in files_in_sub:
                    raw = download_txt_as_text(drive_service, file['id'])
                    if raw:
                        docs.append(raw.strip())
            st.session_state.reference_docs = docs
            st.success(f"Loaded {len(docs)} reference files.")
else:
    st.info("Please select at least one subfolder to load reference documents.")

reference_docs = st.session_state.reference_docs

if reference_docs:
    st.info(f"Currently loaded reference documents: {len(reference_docs)}")
else:
    st.info("No reference documents loaded.")

# -------- Proposal Upload Section ----------
st.subheader("2. Upload Proposal (Your Document)")
prop_file = st.file_uploader(
    "Upload your proposal or template (.txt, .docx, .pdf)", type=["txt", "docx", "pdf"])

proposal_text = ""
if prop_file:
    with st.spinner("Parsing proposal..."):
        proposal_text = parse_uploaded_file(prop_file)
        if len(proposal_text) > PROPOSAL_CHAR_LIMIT:
            st.warning(
                f"Proposal is large ({len(proposal_text)} chars). Only the first {PROPOSAL_CHAR_LIMIT} characters will be used."
            )
        st.info(f"Uploaded proposal has approx. {max(1, len(proposal_text)//5)} words.")

# ------- Quick Prompts Section ------------
st.subheader("3. Query")

if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = None

quick_col1, quick_col2, quick_col3 = st.columns(3)
with quick_col1:
    if st.button("Finance Concurrence"):
        st.session_state.quick_prompt = PRESET_QUERIES["Finance Concurrence"]
with quick_col2:
    if st.button("Payment Proposal"):
        st.session_state.quick_prompt = PRESET_QUERIES["Payment Proposal"]
with quick_col3:
    if st.button("Internal Audit"):
        st.session_state.quick_prompt = PRESET_QUERIES["Internal Audit"]

quick_prompt = st.session_state.quick_prompt

# --------- Language Model Selection ---------
st.subheader("4. Select Language Model")
model_cols = st.columns(len(MODEL_MAP))
for i, (label, model) in enumerate(MODEL_MAP.items()):
    if model_cols[i].button(label):
        st.session_state.selected_model = model
        st.session_state.selected_model_label = label

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-3.5-turbo"
if "selected_model_label" not in st.session_state:
    st.session_state.selected_model_label = "Alpha (provides a brief and concise summary; optimized for fast responses)"

st.success(f"Model selected: {st.session_state.selected_model_label}")
selected_model = st.session_state.selected_model

# ----- Core Template Filling Logic -----------
def fill_template_openai(reference_docs, proposal_text, user_query, model_name):
    # Use up to CONTEXT_CHUNKS reference docs, each capped at CHUNK_CHAR_LIMIT
    context_block = "\n\n".join(d[:CHUNK_CHAR_LIMIT] for d in reference_docs[:CONTEXT_CHUNKS])
    proposal_text_trimmed = proposal_text[:PROPOSAL_CHAR_LIMIT]

    prompt = f"""You are an expert internal auditor.
Using ONLY the following reference guidelines:

{context_block}

Now, per the following instruction and proposal document, generate an audit response (or fill the template as required):

Instruction:
{user_query}

Proposal document content:
{proposal_text_trimmed}
"""

    input_tokens = count_tokens(prompt, model_name)
    if input_tokens > (TOKEN_BUDGET - MAX_RESPONSE_TOKENS):
        st.warning("Prompt too long. Try selecting fewer reference documents or uploading a smaller proposal.")
        return "Error: Token limit exceeded."

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert auditor and policy assistant. You have to give your output like a human, in the user point of view. The user will be using you to do his work"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_RESPONSE_TOKENS,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI API Error: {e}")
        return "An error occurred in generating the response."

# ----- Quick Prompt Execution -----------
if proposal_text and quick_prompt and reference_docs:
    st.info("Quick prompt selected, generating response...")
    output = fill_template_openai(reference_docs, proposal_text, quick_prompt, selected_model)
    st.subheader("Result")
    st.write(output)
    st.download_button("Download response as TXT", output, file_name="audit_response.txt", mime="text/plain")
    st.session_state.quick_prompt = None

# ----- Custom Query Section -----------
st.subheader("5. Custom Query")
if "user_query" not in st.session_state:
    st.session_state.user_query = ""

user_query = st.text_area("Or enter a custom query", value=st.session_state.get("user_query", ""), height=80)
if user_query != st.session_state.get("user_query", ""):
    st.session_state.user_query = user_query

if proposal_text and user_query and reference_docs:
    if st.button("Generate Response"):
        with st.spinner("Working..."):
            output = fill_template_openai(reference_docs, proposal_text, user_query, selected_model)
        st.subheader("Result")
        st.write(output)
        st.download_button("Download response as TXT", output, file_name="audit_response.txt", mime="text/plain")
elif not proposal_text:
    st.info("Upload a proposal document to enable generation.")
elif not user_query and not quick_prompt:
    st.info("Enter a query or use a Quick Prompt to generate the response.")
elif not reference_docs:
    st.info("Please select and load reference documents from Google Drive.")

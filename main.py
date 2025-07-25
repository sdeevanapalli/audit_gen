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

# CHROMADB VECTOR
import chromadb
from chromadb.config import Settings

# ENV CONFIGS AND BASIC SETUP

load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
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
chroma_client = chromadb.Client(Settings())
collection = chroma_client.get_or_create_collection("rulebook_rag")

def count_tokens(text, model="gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
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
        return uploaded_file.read().decode('utf-8', errors='ignore')
    elif fname.endswith('.docx'):
        doc = Document(io.BytesIO(uploaded_file.read()))
        return "\n".join([para.text for para in doc.paragraphs])
    elif fname.endswith('.pdf'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp.close()
            text = extract_pdf_text(tmp.name)
            os.remove(tmp.name)
            return text
    else:
        return ""

def embed_text(text):
    text = text.strip()[:2000]
    resp = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
    )
    return resp.data[0].embedding

def chunk_text(text, max_chars=CHUNK_CHAR_LIMIT):
    paras = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 0]
    chunks, chunk = [], ''
    for para in paras:
        if len(chunk) + len(para) + 2 > max_chars:
            if chunk:
                chunks.append(chunk)
                chunk = ''
        chunk += para + '\n\n'
    if chunk:
        chunks.append(chunk)
    return chunks

def index_drive_files(service, files, subfolder_name):
    docs, metas, ids = [], [], []
    for file in files:
        raw = download_txt_as_text(service, file['id'])
        for ci, chunk in enumerate(chunk_text(raw)):
            docs.append(chunk)
            metas.append({'subfolder': subfolder_name, 'filename': file['name']})
            ids.append(f"{subfolder_name}-{file['name']}-{ci}")
    if docs:
        embeds = [embed_text(doc) for doc in docs]
        collection.add(
            documents=docs,
            embeddings=embeds,
            metadatas=metas,
            ids=ids
        )
    return len(docs)

def retrieve_context(query, top_k=CONTEXT_CHUNKS, max_chunk_chars=CHUNK_CHAR_LIMIT):
    q_embed = embed_text(query)
    results = collection.query(
        query_embeddings=[q_embed],
        n_results=top_k
    )
    docs = [doc[:max_chunk_chars] for doc in results['documents'][0]]
    metas = results['metadatas'][0]
    return docs, metas

def fill_template(proposal_text, user_query, model_name):
    proposal_text_trimmed = proposal_text[:PROPOSAL_CHAR_LIMIT]
    rag_query = user_query + "\n\nProposal:\n" + proposal_text_trimmed
    contexts, metas = retrieve_context(rag_query)
    context_block = "\n\n".join([f"{m['subfolder']}/{m['filename']}:\n{c}" for c, m in zip(contexts, metas)])
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
        st.warning(f"Prompt ({input_tokens} tokens) too long for {model_name}. Trimming context.")
        for i in range(len(contexts)):
            trimmed_contexts = contexts[:i+1]
            block = "\n\n".join([f"{metas[j]['subfolder']}/{metas[j]['filename']}:\n{contexts[j]}" for j in range(i+1)])
            curr_prompt = f"""You are an expert internal auditor.
Using ONLY the following reference guidelines:

{block}

Now, per the following instruction and proposal document, generate an audit response (or fill the template as required):

Instruction:
{user_query}

Proposal document content:
{proposal_text_trimmed}
"""
            if count_tokens(curr_prompt, model_name) < (TOKEN_BUDGET - MAX_RESPONSE_TOKENS):
                context_block = block
                prompt = curr_prompt
                break
        else:
            st.error("Unable to fit context within model limits. Use a shorter proposal and/or shorter prompt.")
            return "Error: Unable to fit input within model token limits."
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an expert auditor and policy assistant. You have to give your output like a human, in the user point of view. The user will be using you to do his work"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=MAX_RESPONSE_TOKENS,
    )
    return response.choices[0].message.content

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("Masthead.png", use_container_width=True)
    st.markdown("<h1 style='text-align: center;'>Internal Audit Officer (IAO)</h1>", unsafe_allow_html=True)

try:
    drive_service = get_drive_service()
except Exception as e:
    st.error(f"Google Drive authentication/setup failed: {e}")
    st.stop()

st.subheader("1. Select Reference Subfolders")
subfolders = list_subfolders(drive_service, DRIVE_MAIN_FOLDER_ID)
if not subfolders:
    st.warning("No subfolders found in project#1.")
    st.stop()
subfolder_names = [f['name'] for f in subfolders]
subfolder_map = {f['name']: f['id'] for f in subfolders}

selected_subfolders = st.multiselect("Choose one or more subfolders", subfolder_names)

if selected_subfolders:
    if st.button(f"Index reference files in: {', '.join(selected_subfolders)}"):
        total_chunks = 0
        with st.spinner("Indexing files for fast search..."):
            for subfolder_name in selected_subfolders:
                files_in_sub = list_txt_in_folder(drive_service, subfolder_map[subfolder_name])
                n_chunks = index_drive_files(drive_service, files_in_sub, subfolder_name)
                total_chunks += n_chunks
            st.success(f"Indexed {len(selected_subfolders)} subfolders, total {total_chunks} chunks added to RAG DB.")
else:
    st.info("Please select at least one subfolder to index.")

st.subheader("2. Upload Proposal (Your Document)")
prop_file = st.file_uploader(
    "Upload your proposal or template (.txt, .docx, .pdf)", type=["txt", "docx", "pdf"]
)
proposal_text = ""
if prop_file:
    with st.spinner("Parsing proposal..."):
        proposal_text = parse_uploaded_file(prop_file)
        if len(proposal_text) > PROPOSAL_CHAR_LIMIT:
            st.warning(
                f"Proposal is large ({len(proposal_text)} chars). Only the first {PROPOSAL_CHAR_LIMIT} characters will be used."
            )
        st.info(f"Uploaded proposal has approx. {len(proposal_text)//5} words.")
st.subheader("3. Query")
quick_prompt = None
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Finance Concurrence"):
        quick_prompt = PRESET_QUERIES["Finance Concurrence"]
with col2:
    if st.button("Payment Proposal"):
        quick_prompt = PRESET_QUERIES["Payment Proposal"]
with col3:
    if st.button("Internal Audit"):
        quick_prompt = PRESET_QUERIES["Internal Audit"]


st.subheader("4. Select Language Model")
cols = st.columns(len(MODEL_MAP))
for i, (label, model) in enumerate(MODEL_MAP.items()):
    if cols[i].button(label):
        st.session_state.selected_model = model
        st.session_state.selected_model_label = label

# Ensure default model and label are set
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-3.5-turbo"
if "selected_model_label" not in st.session_state:
    st.session_state.selected_model_label = "Alpha (provides a brief and concise summary; optimized for fast responses)"

# Display current model selection
st.success(f"Model selected: {st.session_state.selected_model_label}")
selected_model = st.session_state.selected_model  # Always available after this point

# Quick Prompt Execution
if proposal_text and quick_prompt:
    st.info("Quick prompt selected, generating response...")
    output = fill_template(proposal_text, quick_prompt, selected_model)
    st.subheader("Result")
    st.write(output)
    st.download_button("Download response as TXT", output, file_name="audit_response.txt", mime="text/plain")

st.subheader("5. Custom Query")
user_query = st.text_area("Or enter a custom query", value=st.session_state.get("user_query", ""), height=80)

if proposal_text and user_query:
    if st.button("Generate Response"):
        with st.spinner("Working..."):
            output = fill_template(proposal_text, user_query, selected_model)
        st.subheader("Result")
        st.write(output)
        st.download_button("Download response as TXT", output, file_name="audit_response.txt", mime="text/plain")
elif not proposal_text:
    st.info("Upload a proposal document to enable generation.")
elif not user_query:
    st.info("Enter a query or use a Quick Prompt to generate the response.")

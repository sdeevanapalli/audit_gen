import os
import io
import tempfile
import atexit
import json
import streamlit as st
import numpy as np

from openai import OpenAI
from docx import Document
from pdfminer.high_level import extract_text as extract_pdf_text
from dotenv import load_dotenv
import tiktoken

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ENV CONFIG AND SETUP
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

CONTEXT_CHUNKS = 10
CHUNK_CHAR_LIMIT = 1000
PROPOSAL_CHAR_LIMIT = 2200
TOKEN_BUDGET = 8000
MAX_RESPONSE_TOKENS = 1024
SUMMARY_MAX_TOKENS = 256

client = OpenAI(api_key=API_KEY)


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

# RAG WORK - SEMANTIC SEARCH BASIS
def chunk_documents(reference_docs, chunk_size=CHUNK_CHAR_LIMIT):
    chunks = []
    for doc_index, doc in enumerate(reference_docs):
        for i in range(0, len(doc), chunk_size):
            chunk = doc[i:i+chunk_size]
            if chunk.strip():
                chunks.append({'text': chunk, 'doc_index': doc_index, 'chunk_start': i})
    return chunks

def get_embeddings_for_chunks(chunks):
    texts = [chunk['text'] for chunk in chunks]
    max_batch = 96
    out = []
    for i in range(0, len(texts), max_batch):
        response = client.embeddings.create(
            input=texts[i:i+max_batch],
            model="text-embedding-3-small"
        )
        emb = [np.array(d.embedding) for d in response.data]
        out.extend(emb)
    return out

def embedding_for_query(query):
    response = client.embeddings.create(
        input=[query],
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding)

def retrieve_relevant_chunks(reference_docs, user_query, k=CONTEXT_CHUNKS):

    if "rag_ref_docs_copy" not in st.session_state or st.session_state.rag_ref_docs_copy != reference_docs:
        chunks = chunk_documents(reference_docs)
        st.session_state.rag_chunks = chunks
        st.session_state.rag_chunks_embeddings = get_embeddings_for_chunks(chunks) if chunks else []
        st.session_state.rag_ref_docs_copy = list(reference_docs)

    if not st.session_state.rag_chunks:
        return []

    query_emb = embedding_for_query(user_query)
    chunk_embs = st.session_state.rag_chunks_embeddings
    sims = [float(np.dot(query_emb, c) / (np.linalg.norm(query_emb) * np.linalg.norm(c) + 1e-8)) for c in chunk_embs]
    idxs = np.argsort(sims)[::-1][:k]
    relevant_chunks = [st.session_state.rag_chunks[i]['text'] for i in idxs]
    return relevant_chunks

def assemble_context(reference_docs, user_query, k=CONTEXT_CHUNKS):
    relevant_chunks = retrieve_relevant_chunks(reference_docs, user_query, k=k)
    context_block = "\n\n".join(relevant_chunks)
    return context_block

def run_model(context_block, proposal_block, user_query, model_name):
    use_proposal = (
        proposal_block
        and ("proposal" in user_query.lower() or "uploaded document" in user_query.lower())
    )
    prompt = f"""
You are an expert policy assistant and internal auditor.
Using ONLY the text below as your source, provide a well-organized, friendly and helpful answer to the user's question.

- Synthesize and summarize across all the relevant information.
- Structure your answer in clear bullet points, sections, or paragraphs (as appropriate).
- Use **bold** for section headings, emoji if helpful, and make the answer easy to read for a non-expert.
- When your answer uses information from a specific statute, ordinance, section, or reference document, you MUST clearly mention its name and (if available) section/number (e.g., Statute 7A, Ordinance 5, Section 14).
- Use parenthesis or square brackets for references, e.g., (Statute 7A), [Ordinance No. 3], etc.
- DO NOT just copy-paste the raw statute—write in your own words.

Reference Documents:
{context_block}
""" + (
        f"\nProposal document:\n{proposal_block}\n" if use_proposal else ""
    ) + f"""
User Question:
{user_query}

If the answer is not found in the provided context, respond: "The answer is not present in the provided references." Otherwise, answer fully, using a friendly, complete, and helpful style.
"""
    input_tokens = count_tokens(prompt, model_name)
    if input_tokens > (TOKEN_BUDGET - MAX_RESPONSE_TOKENS):
        st.warning("Prompt too long. Try selecting fewer reference documents.")
        return "Error: Token limit exceeded."
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert auditor and policy assistant. Your job is to help the user by providing high-quality, easy to understand, fully structured answers using ONLY the context supplied."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_RESPONSE_TOKENS,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI API Error: {e}")
        return "An error occurred in generating the response."


def make_summary(full_answer, model_name):
    summary_prompt = f"""
Summarize the following answer in 2-4 lines for a 'Summary (TL;DR)' box at the end of a report. Focus on only the essential points and avoid repetition. Write in clear plain English.

Answer:
{full_answer}

TL;DR:
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant who summarizes text for users in short, plain language for non-experts."},
                {"role": "user", "content": summary_prompt}
            ],
            max_tokens=SUMMARY_MAX_TOKENS,
        )
        return response.choices[0].message.content
    except Exception:
        return "Could not generate summary."

# STREAMLIT UI

colx1, colx2, colx3 = st.columns([1,2,1])
with colx2:
    st.image("Masthead.png", use_container_width=True)
    st.markdown("<h1 style='text-align: center;'>Internal Audit Officer (IAO)</h1>", unsafe_allow_html=True)

try:
    drive_service = get_drive_service()
except Exception as e:
    st.error(f"Google Drive authentication/setup failed: {e}")
    st.stop()

# SUBFOLDER
st.subheader("Select Reference Subfolders")
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
            # Reset RAG cache!
            for k in ['rag_chunks','rag_chunks_embeddings','rag_ref_docs_copy']:
                if k in st.session_state: del st.session_state[k]
            st.success(f"Loaded {len(docs)} reference files.")
else:
    st.info("Please select at least one subfolder to load reference documents.")

reference_docs = st.session_state.reference_docs

if reference_docs:
    st.info(f"Currently loaded reference documents: {len(reference_docs)}")
else:
    st.info("No reference documents loaded.")

# MODE SELECTION
st.subheader("Select Mode")
mode = st.radio(
    "Choose mode",
    ["Report/Template Generation", "Query Answering"],
    horizontal=True
)

# PROPOSAL DOC UPLOAD
proposal_text = ""
if mode == "Report/Template Generation":
    st.subheader("Upload Proposal (Optional)")
    prop_file = st.file_uploader(
        "Upload your proposal or template (.txt, .docx, .pdf)",
        type=["txt", "docx", "pdf"]
    )
    if prop_file:
        with st.spinner("Parsing proposal..."):
            proposal_text = parse_uploaded_file(prop_file)
            if len(proposal_text) > PROPOSAL_CHAR_LIMIT:
                st.warning(
                    f"Proposal is large ({len(proposal_text)} chars). Only the first {PROPOSAL_CHAR_LIMIT} characters will be used."
                )
            st.info(f"Uploaded proposal has approx. {max(1, len(proposal_text)//5)} words.")

# QUICK PROMPTS
if mode == "Report/Template Generation":
    st.subheader("Quick Prompts")
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

# MODEL SELECTION
st.subheader("Select Model")
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

# SUBMISSION

if mode == "Report/Template Generation":
    st.subheader("6. Report/Template Query")
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    user_query = st.text_area("Enter a query, or select a Quick Prompt above", value=st.session_state.get("user_query", ""), height=80)

    if user_query != st.session_state.get("user_query", ""):
        st.session_state.user_query = user_query

    submit_custom_query = st.button("Submit")

    # QUICK PROMPT
    used_query = user_query
    if st.session_state.get("quick_prompt"):
        used_query = st.session_state.quick_prompt

    if (submit_custom_query or st.session_state.get("quick_prompt")) and used_query:
        if reference_docs:
            with st.spinner("Fetching relevant context and generating report..."):
                context_block = assemble_context(reference_docs, used_query)
                output = run_model(context_block, proposal_text, used_query, selected_model)
                summary = make_summary(output, selected_model)
            st.subheader("Result")
            st.write(output)
            st.markdown(f"---\n<b>Summary (TL;DR):</b><br>{summary}", unsafe_allow_html=True)
            st.download_button("Download response as TXT", output + "\n\nSummary (TL;DR):\n" + summary,
                               file_name="audit_response.txt", mime="text/plain")
            st.session_state.quick_prompt = None
        else:
            st.info("Please select and load reference documents from Google Drive.")

elif mode == "Query Answering":
    st.subheader("Ask a Question (No proposal necessary)")
    if "user_query_qa" not in st.session_state:
        st.session_state.user_query_qa = ""
    user_query_qa = st.text_area("Ask a policy/guideline question", value=st.session_state.get("user_query_qa", ""), height=80)
    submit_query_qa = st.button("Get Answer")

    if submit_query_qa and user_query_qa:
        if reference_docs:
            with st.spinner("Searching references and answering..."):
                context_block = assemble_context(reference_docs, user_query_qa)
                output = run_model(context_block, None, user_query_qa, selected_model)
                summary = make_summary(output, selected_model)
            st.subheader("Answer")
            st.write(output)
            st.markdown(f"---\n<b>Summary (TL;DR):</b><br>{summary}", unsafe_allow_html=True)
            st.download_button("Download answer as TXT", output + "\n\nSummary (TL;DR):\n" + summary,
                               file_name="query_answer.txt", mime="text/plain")
        else:
            st.info("Please select and load reference documents from Google Drive.")

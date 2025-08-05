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
import time

# ENV CONFIG AND SETUP
try:
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")
    SERVICE_ACCOUNT_JSON = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
    DRIVE_MAIN_FOLDER_ID = os.getenv('GOOGLE_DRIVE_FOLDER_ID')
    if not API_KEY or not SERVICE_ACCOUNT_JSON or not DRIVE_MAIN_FOLDER_ID:
        st.error("One or more required environment variables are missing. Please check your .env or secret configs before starting the app.")
        st.stop()
except Exception as e:
    st.error(f"Failed loading environment vars: {e}")
    st.stop()

try:
    st.set_page_config(page_title="Internal Audit Officer", layout="wide")
except Exception:
    pass

MODEL_MAP = {
    "Alpha (provides a brief and concise summary; optimized for fast responses)": "gpt-3.5-turbo",
    "Beta (offers a balanced output combining brevity and detail)": "gpt-4",
    "Gamma (delivers comprehensive and detailed analysis; may take longer to respond)": "gpt-4o"
}

PRESET_QUERIES = {
    "Finance Concurrence": "Please examine the uploaded document in accordance with applicable Government of India financial rules and procedures, including the General Financial Rules (GFR) and any other relevant guidelines provided via the linked Google Drive. Based on this review, provide a finance concurrence report, citing all applicable provisions and ensuring compliance.",
    "Payment Proposal": "Please prepare a payment proposal for the uploaded document using the format outlined in the file attached if present. The proposal must strictly follow the applicable government guidelines and provisions of the General Financial Rules (GFR) available in the linked Google Drive. Ensure that all required details, justifications, and calculations are provided clearly.",
    "Internal Audit": "Please draft an internal audit report for the uploaded proposal based on the procedures and templates outlined in the internal audit manual. The report must adhere to all relevant audit standards, check compliance with the GFR and applicable guidelines available on the linked Google Drive, and flag any deviations, observations, or required follow-ups."
}

CONTEXT_CHUNKS = 40
CHUNK_CHAR_LIMIT = 2000
PROPOSAL_CHAR_LIMIT = 30000
TOKEN_BUDGET = 125000 #gpt 4o limit is 128k
MAX_RESPONSE_TOKENS = 2500
SUMMARY_MAX_TOKENS = 512

# OpenAI client initialisation
try:
    client = OpenAI(api_key=API_KEY)
except Exception as e:
    st.error(f"Failed to initialise OpenAI client: {e}")
    st.stop()

def count_tokens(text, model="gpt-4o"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            st.error(f"Token encoding error: {e}")
            return len(text)
    try:
        return len(enc.encode(text))
    except Exception as e:
        st.error(f"Token count error: {e}")
        return len(text)

def get_drive_service():
    try:
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
    except Exception as e:
        st.error(f"Google Drive service setup failed: {e}")
        raise

def list_subfolders(service, parent_id):
    try:
        results = []
        page_token = None
        while True:
            resp = service.files().list(
                q=f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
                fields="nextPageToken, files(id, name)", pageToken=page_token
            ).execute()
            results.extend(resp.get('files', []))
            page_token = resp.get('nextPageToken', None)
            if not page_token:
                break
        return results
    except Exception as e:
        st.error(f"Could not list subfolders: {e}")
        return []

def list_txt_in_folder(service, folder_id):
    try:
        results = []
        page_token = None
        while True:
            resp = service.files().list(
                q=f"'{folder_id}' in parents and mimeType='text/plain' and trashed=false",
                pageSize=500,
                fields="nextPageToken, files(id, name)", pageToken=page_token
            ).execute()
            results.extend(resp.get('files', []))
            page_token = resp.get('nextPageToken', None)
            if not page_token:
                break
        return results
    except Exception as e:
        st.error(f"Could not list TXT files in folder: {e}")
        return []

def download_txt_as_text(service, file_id):
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            try:
                _, done = downloader.next_chunk()
            except Exception as e:
                st.error(f"Download chunk error: {e}")
                break
        fh.seek(0)
        return fh.read().decode('utf-8', errors='ignore')
    except Exception as e:
        st.error(f"Failed to download text from drive: {e}")
        return ""

def parse_uploaded_file(uploaded_file):
    try:
        fname = uploaded_file.name.lower()
        if fname.endswith('.txt'):
            try:
                bytes_content = uploaded_file.read()
                if isinstance(bytes_content, bytes):
                    return bytes_content.decode('utf-8', errors='ignore')
                else:
                    return str(bytes_content)
            except Exception:
                st.error("Failed to decode TXT file. Please use UTF-8 encoding.")
                return ""
        elif fname.endswith('.docx'):
            try:
                doc = Document(io.BytesIO(uploaded_file.read()))
                return "\n".join([para.text for para in doc.paragraphs])
            except Exception:
                st.error("Failed to parse DOCX. File may be corrupt or wrong format.")
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
                st.error("Failed to parse PDF. The file may be encrypted or corrupt.")
                return ""
        else:
            st.warning("Unsupported file format. Only .txt, .docx, .pdf allowed.")
            return ""
    except Exception as e:
        st.error(f"Could not parse uploaded file: {e}")
        return ""

def chunk_documents(reference_docs, chunk_size=CHUNK_CHAR_LIMIT):
    try:
        chunks = []
        for doc_index, doc in enumerate(reference_docs):
            doc = (doc or "").strip()
            if not doc:
                continue
            for i in range(0, len(doc), chunk_size):
                chunk = doc[i:i + chunk_size]
                if chunk.strip():
                    chunks.append({'text': chunk, 'doc_index': doc_index, 'chunk_start': i})
        return chunks
    except Exception as e:
        st.error(f"Chunking documents failed: {e}")
        return []

def safe_openai_call(fn, *args, retries=3, **kwargs):
    """Safe OpenAI call with retries and error messaging."""
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt < retries-1:
                time.sleep(2 ** attempt)
            else:
                st.error(f"OpenAI API Error: {e}")
                return None

def get_embeddings_for_chunks(chunks):
    try:
        texts = [chunk['text'] for chunk in chunks]
        max_batch = 96
        out = []
        for i in range(0, len(texts), max_batch):
            response = safe_openai_call(client.embeddings.create,
                input=texts[i:i + max_batch],
                model="text-embedding-3-small"
            )
            if response is None or not hasattr(response, "data"):
                st.error("Failed to compute context embeddings via OpenAI.")
                return []
            emb = [np.array(d.embedding) for d in response.data]
            out.extend(emb)
        return out
    except Exception as e:
        st.error(f"Embedding for chunks failed: {e}")
        return []

def embedding_for_query(query):
    try:
        response = safe_openai_call(client.embeddings.create,
            input=[query],
            model="text-embedding-3-small"
        )
        if response is None or not hasattr(response, "data"):
            st.error("Failed to compute embedding for query via OpenAI.")
            return np.zeros(1536)
        return np.array(response.data[0].embedding)
    except Exception as e:
        st.error(f"Embedding for query failed: {e}")
        return np.zeros(1536)

def retrieve_relevant_chunks(reference_docs, user_query, k=CONTEXT_CHUNKS):
    try:
        # Hash and compare, better session cache
        ref_hash = hash(tuple(reference_docs))
        if "rag_ref_docs_hash" not in st.session_state or st.session_state.rag_ref_docs_hash != ref_hash:
            st.session_state.rag_chunks = chunk_documents(reference_docs)
            st.session_state.rag_chunks_embeddings = get_embeddings_for_chunks(st.session_state.rag_chunks) if st.session_state.rag_chunks else []
            st.session_state.rag_ref_docs_hash = ref_hash
        if not st.session_state.rag_chunks:
            return []
        query_emb = embedding_for_query(user_query)
        chunk_embs = st.session_state.rag_chunks_embeddings
        if not chunk_embs:
            st.warning("No embeddings for context. Try reloading reference docs.")
            return []
        sims = []
        for c in chunk_embs:
            try:
                sim = float(np.dot(query_emb, c) / (np.linalg.norm(query_emb) * np.linalg.norm(c) + 1e-8))
            except Exception:
                sim = 0.0
            sims.append(sim)
        idxs = np.argsort(sims)[::-1][:k]
        relevant_chunks = [st.session_state.rag_chunks[i]['text'] for i in idxs]
        return relevant_chunks
    except Exception as e:
        st.error(f"Semantic search failed: {e}")
        return []

def assemble_context(reference_docs, user_query, k=CONTEXT_CHUNKS):
    try:
        relevant_chunks = retrieve_relevant_chunks(reference_docs, user_query, k=k)
        if not relevant_chunks:
            st.warning("No relevant reference documents found for this query.")
        context_block = "\n\n".join(relevant_chunks)
        return context_block
    except Exception as e:
        st.error(f"Assembling context failed: {e}")
        return ""

def run_model(context_block, proposal_block, user_query, model_name):
    try:
        use_proposal = (
            proposal_block
            and ("proposal" in user_query.lower() or "uploaded document" in user_query.lower())
        )
        prompt = f"""
You are an expert internal auditor and financial policy assistant. Using only the content provided (uploaded files, templates, and references), generate a clear, structured, and accurate report. Do not use external knowledge.
1. Tone & Purpose
Keep your tone professional, clear, and informative.
Reports should be understandable to non-experts but grounded in policy.
2. Structure
Use clear sections with bold titles, such as:
Summary: What the document is and what you’re evaluating.
Compliance Review: List applicable rules (e.g., Rule 136, GFR 2017) and check adherence.
Observations: Note missing data, errors, or issues.
Conclusion/Recommendation: State whether the proposal is acceptable, needs correction, or is non-compliant.
3. Citations
When referencing policies:
Mention exact rule names and numbers.
Use parentheses or brackets for citations: e.g., (Rule 47, GFR 2017), [Audit Manual, Sec. 3.2].
4. Clarity & Formatting
Use bullet points where needed.
Avoid long paragraphs.
Be concise but complete.
5. Restrictions
Do not hallucinate information or rules.
Do not say “as per guidelines” without specifying the document.
Do not copy large sections of text—summarize instead.

Reference Documents:
{context_block}
""" + (
            f"\nProposal document:\n{proposal_block}\n" if use_proposal else ""
        ) + f"""
User Question:
{user_query}

If the answer is not found in the provided context, respond: "The answer is not present in the provided references." Otherwise, answer fully, using a friendly, complete, professional and helpful style.
"""
        input_tokens = count_tokens(prompt, model_name)
        if input_tokens > (TOKEN_BUDGET - MAX_RESPONSE_TOKENS):
            # attempt truncation
            orig_chunks = context_block.split("\n\n")
            while input_tokens > (TOKEN_BUDGET - MAX_RESPONSE_TOKENS) and len(orig_chunks) > 1:
                orig_chunks = orig_chunks[:-1]
                context_block_new = "\n\n".join(orig_chunks)
                prompt = prompt.replace(context_block, context_block_new)
                context_block = context_block_new
                input_tokens = count_tokens(prompt, model_name)
            if input_tokens > (TOKEN_BUDGET - MAX_RESPONSE_TOKENS):
                st.warning("Prompt too long. Try selecting fewer reference documents.")
                return "Error: Token limit exceeded."
        response = safe_openai_call(client.chat.completions.create,
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert auditor and policy assistant. Your job is to help the user by providing high-quality, easy to understand, fully structured answers using ONLY the context supplied."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_RESPONSE_TOKENS,
        )
        if not response or not hasattr(response, "choices"):
            st.error("No response from OpenAI API.")
            return "An error occurred in generating the response."
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Prompt building or model error: {e}")
        return "Model run error."

def make_summary(full_answer, model_name):
    summary_prompt = f"""
Summarize the following answer in 2-4 lines for a 'Summary (TL;DR)' box at the end of a report. Focus on only the essential points and avoid repetition. Write in clear plain English.

Answer:
{full_answer}

TL;DR:
"""
    try:
        response = safe_openai_call(client.chat.completions.create,
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant who summarizes text for users in short, plain language for non-experts."},
                {"role": "user", "content": summary_prompt}
            ],
            max_tokens=SUMMARY_MAX_TOKENS,
        )
        if not response or not hasattr(response, "choices"):
            return "Could not generate summary."
        return response.choices[0].message.content
    except Exception:
        return "Could not generate summary."

# STREAMLIT UI

colx1, colx2, colx3 = st.columns([1, 2, 1])
with colx2:
    try:
        st.image("Masthead.png", use_container_width=True)
    except Exception:
        pass
    st.markdown("<h3 style='text-align: center;'>Assistant Internal Audit Officer (A-IAO)</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>An artificial Intelligence based prototype</p>", unsafe_allow_html=True)

    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display: none;}
        .st-emotion-cache-13ln4jf {display: none;} /* Share button */
        .st-emotion-cache-ocqkz7 {display: none;} /* GitHub Star button */
        footer {visibility: hidden !important;}
        [data-testid="stToolbar"] {visibility: hidden !important;}
        </style>
    """
    try:
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)
        st.markdown("""
            <style>
            .block-container { padding-top: 1rem !important; }
            .main { padding-top: 0rem !important; }
            </style>
        """, unsafe_allow_html=True)
    except Exception:
        pass

# DRIVE SETUP
try:
    drive_service = get_drive_service()
except Exception as e:
    st.error(f"Google Drive authentication/setup failed: {e}")
    st.stop()

# SUBFOLDER
try:
    st.subheader("Select Reference Subfolders")
    subfolders = list_subfolders(drive_service, DRIVE_MAIN_FOLDER_ID)
    if not subfolders:
        st.warning("No subfolders found in project.")
        st.stop()
    subfolder_names = [f['name'] for f in subfolders]
    subfolder_map = {f['name']: f['id'] for f in subfolders}
except Exception as e:
    st.error(f"Listing subfolders failed: {e}")
    st.stop()

try:
    selected_subfolders = st.multiselect("Choose one or more subfolders", subfolder_names)
except Exception:
    selected_subfolders = []

if "reference_docs" not in st.session_state:
    st.session_state.reference_docs = []

if selected_subfolders:
    if st.button(f"Fetch and combine files from: {', '.join(selected_subfolders)}"):
        docs = []
        with st.spinner("Fetching reference files..."):
            try:
                for subfolder_name in selected_subfolders:
                    files_in_sub = list_txt_in_folder(drive_service, subfolder_map[subfolder_name])
                    for file in files_in_sub:
                        raw = download_txt_as_text(drive_service, file['id'])
                        if raw:
                            docs.append(raw.strip())
                st.session_state.reference_docs = docs
                # Reset RAG cache!
                for k in ['rag_chunks', 'rag_chunks_embeddings', 'rag_ref_docs_hash']:
                    if k in st.session_state: del st.session_state[k]
                st.success(f"Loaded {len(docs)} reference files.")
            except Exception as e:
                st.error(f"Reference document fetching error: {e}")
else:
    st.info("Please select at least one subfolder to load reference documents.")

reference_docs = st.session_state.reference_docs

try:
    if reference_docs:
        st.info(f"Currently loaded reference documents: {len(reference_docs)}")
    else:
        st.info("No reference documents loaded.")
except Exception:
    pass

# MODE SELECTION
try:
    st.subheader("Select Mode")
    mode = st.radio(
        "Choose mode",
        ["Report/Template Generation", "Query Answering"],
        horizontal=True
    )
except Exception:
    mode = "Report/Template Generation"

# PROPOSAL DOC UPLOAD
proposal_text = ""
if mode == "Report/Template Generation":
    st.subheader("Upload Proposal (Optional)")
    try:
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
                    proposal_text = proposal_text[:PROPOSAL_CHAR_LIMIT]
                st.info(f"Uploaded proposal has approx. {max(1, len(proposal_text)//5)} words.")
    except Exception as e:
        st.error(f"Proposal parsing/upload failed: {e}")

# QUICK PROMPTS
if mode == "Report/Template Generation":
    st.subheader("Quick Prompts")
    if "quick_prompt" not in st.session_state:
        st.session_state.quick_prompt = None

    quick_col1, quick_col2, quick_col3 = st.columns(3)
    try:
        with quick_col1:
            if st.button("Finance Concurrence"):
                st.session_state.quick_prompt = PRESET_QUERIES["Finance Concurrence"]
        with quick_col2:
            if st.button("Payment Proposal"):
                st.session_state.quick_prompt = PRESET_QUERIES["Payment Proposal"]
        with quick_col3:
            if st.button("Internal Audit"):
                st.session_state.quick_prompt = PRESET_QUERIES["Internal Audit"]
    except Exception:
        pass

    quick_prompt = st.session_state.quick_prompt

# MODEL SELECTION
st.subheader("Select Model")
try:
    model_cols = st.columns(len(MODEL_MAP))
    for i, (label, model) in enumerate(MODEL_MAP.items()):
        if model_cols[i].button(label):
            st.session_state.selected_model = model
            st.session_state.selected_model_label = label
except Exception:
    st.session_state.selected_model = "gpt-3.5-turbo"
    st.session_state.selected_model_label = "Alpha (provides a brief and concise summary; optimized for fast responses)"

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-3.5-turbo"
if "selected_model_label" not in st.session_state:
    st.session_state.selected_model_label = "Alpha (provides a brief and concise summary; optimized for fast responses)"

st.success(f"Model selected: {st.session_state.selected_model_label}")
selected_model = st.session_state.selected_model

# SUBMISSION

if mode == "Report/Template Generation":
    st.subheader("Report/Template Query")
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    try:
        user_query = st.text_area("Enter a query, or select a Quick Prompt above", value=st.session_state.get("user_query", ""), height=80)
        if user_query != st.session_state.get("user_query", ""):
            st.session_state.user_query = user_query
    except Exception:
        user_query = ""

    submit_custom_query = st.button("Submit")

    # QUICK PROMPT
    used_query = user_query
    if st.session_state.get("quick_prompt"):
        used_query = st.session_state.quick_prompt

    if (submit_custom_query or st.session_state.get("quick_prompt")) and used_query:
        if reference_docs:
            with st.spinner("Fetching relevant context and generating report..."):
                try:
                    context_block = assemble_context(reference_docs, used_query)
                    output = run_model(context_block, proposal_text, used_query, selected_model)
                    summary = make_summary(output, selected_model)
                except Exception as e:
                    st.error(f"Report/model error: {e}")
                    output = "Error"
                    summary = "Error"
            st.subheader("Result")
            try:
                st.write(output)
                st.markdown(f"---\n<b>Summary (TL;DR):</b><br>{summary}", unsafe_allow_html=True)
                st.download_button("Download response as TXT", output + "\n\nSummary (TL;DR):\n" + summary,
                                   file_name="audit_response.txt", mime="text/plain")
            except Exception as e:
                st.error(f"Displaying answer or download failed: {e}")
            st.session_state.quick_prompt = None
        else:
            st.info("Please select and load reference documents from Google Drive.")

elif mode == "Query Answering":
    st.subheader("Ask a Question (No proposal necessary)")
    if "user_query_qa" not in st.session_state:
        st.session_state.user_query_qa = ""
    try:
        user_query_qa = st.text_area("Ask a policy/guideline question",
                                     value=st.session_state.get("user_query_qa", ""), height=80)
        submit_query_qa = st.button("Get Answer")
    except Exception:
        user_query_qa = ""
        submit_query_qa = False

    if submit_query_qa and user_query_qa:
        if reference_docs:
            with st.spinner("Searching references and answering..."):
                try:
                    context_block = assemble_context(reference_docs, user_query_qa)
                    output = run_model(context_block, None, user_query_qa, selected_model)
                    summary = make_summary(output, selected_model)
                except Exception as e:
                    st.error(f"QA/model error: {e}")
                    output = "Error"
                    summary = "Error"
            st.subheader("Answer")
            try:
                st.write(output)
                st.markdown(f"---\n<b>Summary (TL;DR):</b><br>{summary}", unsafe_allow_html=True)
                st.download_button("Download answer as TXT", output + "\n\nSummary (TL;DR):\n" + summary,
                                   file_name="query_answer.txt", mime="text/plain")
            except Exception as e:
                st.error(f"Displaying answer or download failed: {e}")
        else:
            st.info("Please select and load reference documents from Google Drive.")

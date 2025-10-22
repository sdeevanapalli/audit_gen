import os
import io
import tempfile
import atexit
import json
import streamlit as st
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from docx import Document
from pdfminer.high_level import extract_text as extract_pdf_text
from dotenv import load_dotenv
import tiktoken

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import pickle
import time


# ENV CONFIG AND SETUP
@st.cache_data
def load_environment() -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """Load and validate environment variables for OpenAI and Drive folder id.

    Note: service account credentials are intentionally NOT required here because
    they may come from Streamlit `st.secrets`, a local file, or one of several
    environment variables. Use `get_service_account_info()` to obtain the
    parsed credentials (dict) in a flexible way.
    """
    try:
        load_dotenv()
        # Prefer environment variables, but also accept Streamlit secrets (for prod)
        API_KEY = os.getenv("OPENAI_API_KEY") or None
        DRIVE_MAIN_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID") or None

        # If running on Streamlit (prod), allow secrets with lowercase keys too
        try:
            if not API_KEY and "openai_api_key" in st.secrets:
                API_KEY = st.secrets.get("openai_api_key")
            if not API_KEY and "OPENAI_API_KEY" in st.secrets:
                API_KEY = st.secrets.get("OPENAI_API_KEY")

            if not DRIVE_MAIN_FOLDER_ID and "google_drive_folder_id" in st.secrets:
                DRIVE_MAIN_FOLDER_ID = st.secrets.get("google_drive_folder_id")
            if not DRIVE_MAIN_FOLDER_ID and "GOOGLE_DRIVE_FOLDER_ID" in st.secrets:
                DRIVE_MAIN_FOLDER_ID = st.secrets.get("GOOGLE_DRIVE_FOLDER_ID")
        except Exception:
            # st may not be available in some contexts; ignore if so
            pass

        missing_vars = [
            var
            for var, val in zip(
                ["OPENAI_API_KEY", "GOOGLE_DRIVE_FOLDER_ID"],
                [API_KEY, DRIVE_MAIN_FOLDER_ID],
            )
            if not val
        ]

        if missing_vars:
            return None, f"Missing environment variables: {', '.join(missing_vars)}"

        return {
            "API_KEY": API_KEY,
            "DRIVE_MAIN_FOLDER_ID": DRIVE_MAIN_FOLDER_ID,
        }, None
    except Exception as e:
        return None, f"Failed to load environment variables: {str(e)}"


# Load environment variables
env_vars, env_error = load_environment()
if env_error:
    st.error(env_error)
    st.stop()

API_KEY = env_vars["API_KEY"]
DRIVE_MAIN_FOLDER_ID = env_vars["DRIVE_MAIN_FOLDER_ID"]


def get_service_account_info() -> Optional[dict]:
    """Return parsed service-account JSON as a dict.

    Loading order (most preferred first):
    - Streamlit secrets: `st.secrets["google_service_account"]` (can be a table/dict or a JSON string)
    - Environment variable `SERVICE_ACCOUNT_FILE` (path to a json file)
    - Environment variable `SERVICE_ACCOUNT_JSON` or `GOOGLE_SERVICE_ACCOUNT_JSON` (JSON string)
    - Local file `service-account.json` in the repo cwd

    Returns None when no credentials are found. Caller should raise a helpful
    error if credentials are required.
    """
    # 1) Streamlit secrets (production - Streamlit Cloud)
    try:
        if "google_service_account" in st.secrets:
            info = st.secrets["google_service_account"]
            # st.secrets can store tables (dict-like) or strings
            if isinstance(info, str):
                return json.loads(info)
            return dict(info)
    except Exception:
        # don't fail hard here; fall back to env/local methods
        pass

    # 2) Local environment (.env) or system env
    load_dotenv()
    sa_file = os.getenv("SERVICE_ACCOUNT_FILE") or os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
    sa_json = os.getenv("SERVICE_ACCOUNT_JSON") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

    if sa_file and os.path.exists(sa_file):
        try:
            with open(sa_file, "r") as fh:
                return json.load(fh)
        except Exception:
            raise ValueError(f"Could not read service account file at {sa_file}")

    if sa_json:
        try:
            return json.loads(sa_json)
        except Exception:
            raise ValueError("SERVICE_ACCOUNT_JSON / GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON")

    # 3) local default filename fallback
    local_path = os.path.join(os.getcwd(), "service-account.json")
    if os.path.exists(local_path):
        try:
            with open(local_path, "r") as fh:
                return json.load(fh)
        except Exception:
            raise ValueError("Found local service-account.json but could not parse it")

    # Nothing found
    return None

# Model configurations
MODEL_MAP = {
    "Alpha - Fastest": "chatgpt-4o-latest",
    "Beta - Advanced reasoning & speed": "gpt-5",
    "Gamma - Large input capacity, detailed tasks": "gpt-4.1",
}

PRESET_QUERIES = {
    "Finance Concurrence": "Please examine the uploaded document in accordance with applicable Government of India financial rules and procedures, including the General Financial Rules (GFR) and any other relevant guidelines provided via the linked Google Drive. Based on this review, provide a finance concurrence report, citing all applicable provisions and ensuring compliance.",
    "Payment Proposal": "Please prepare a payment proposal for the uploaded document using the format outlined in the file attached if present. The proposal must strictly follow the applicable government guidelines and provisions of the General Financial Rules (GFR) available in the linked Google Drive. Ensure that all required details, justifications, and calculations are provided clearly.",
    "Internal Audit": "Please draft an internal audit report for the uploaded proposal based on the procedures and templates outlined in the internal audit manual. Please critically analyze and add weaknesses and strengths. Humanize English. The report must adhere to all relevant audit standards, check compliance with the GFR and applicable guidelines available on the linked Google Drive, and flag any deviations, observations, or required follow-ups.",
}

MODEL_CONFIGS = {
    "chatgpt-4o-latest": {
        "CONTEXT_CHUNKS": 8,  # Fits under 30k TPM
        "CHUNK_CHAR_LIMIT": 2000,  # Smaller per chunk
        "PROPOSAL_CHAR_LIMIT": 30000,
        "TOKEN_BUDGET": 128000,  # Model max, but our call stays < 30k
        "MAX_RESPONSE_TOKENS": 4000,  # Leaves room for input within TPM
        "SUMMARY_MAX_TOKENS": 1000,
    },
    "gpt-5": {
        "CONTEXT_CHUNKS": 8,
        "CHUNK_CHAR_LIMIT": 2000,
        "PROPOSAL_CHAR_LIMIT": 30000,
        "TOKEN_BUDGET": 400000,
        "MAX_RESPONSE_TOKENS": 4000,  # Keep total < 30k TPM
        "SUMMARY_MAX_TOKENS": 1000,
    },
    "gpt-4.1": {
        "CONTEXT_CHUNKS": 8,
        "CHUNK_CHAR_LIMIT": 2000,
        "PROPOSAL_CHAR_LIMIT": 30000,
        "TOKEN_BUDGET": 1000000,
        "MAX_RESPONSE_TOKENS": 4000,
        "SUMMARY_MAX_TOKENS": 1000,
    },
}


def error_handler(msg: str) -> None:
    st.error(
        "An internal error occurred while processing your request. Please try again later."
    )

# Initialize OpenAI client
@st.cache_resource
def get_openai_client() -> Optional[OpenAI]:
    try:
        return OpenAI(api_key=API_KEY)
    except Exception as e:
        error_handler("")
        return None


client = get_openai_client()
if not client:
    st.stop()


def count_tokens(text: str, model: str = "gpt-4.1") -> int:
    """Count tokens in text"""
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            st.warning(
                "An internal error occurred while processing your request. Please try again later."
            )
            return len(text) // 4

    try:
        return len(enc.encode(text))
    except Exception as e:
        st.warning(
            "An internal error occurred while processing your request. Please try again later."
        )
        return len(text) // 4


@st.cache_resource
def get_drive_service() -> Any:
    """Initialize Google Drive service"""
    try:
        # Prefer using any parsed service account info (from st.secrets, env var, or file)
        parsed_json = get_service_account_info()
        if parsed_json:
            if parsed_json.get("private_key"):
                parsed_json["private_key"] = parsed_json["private_key"].replace("\\n", "\n")

            creds = service_account.Credentials.from_service_account_info(
                parsed_json, scopes=["https://www.googleapis.com/auth/drive.readonly"]
            )
            return build("drive", "v3", credentials=creds)

        # Fallback: if a local service-account.json file exists, load from file
        local_path = os.path.join(os.getcwd(), "service-account.json")
        if os.path.exists(local_path):
            creds = service_account.Credentials.from_service_account_file(
                local_path, scopes=["https://www.googleapis.com/auth/drive.readonly"]
            )
            return build("drive", "v3", credentials=creds)

        raise RuntimeError("No service account credentials provided via st.secrets, env, or local file.")
    except Exception as e:
        error_handler("")
        raise


# New: GoogleDriveExtractor and helper to integrate with existing app
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


class GoogleDriveExtractor:
    def __init__(self, auth_method='service_account', credentials_path='credentials.json'):
        self.auth_method = auth_method
        self.credentials_path = credentials_path
        self.service = self._authenticate()

    def _authenticate(self):
        creds = None
        if self.auth_method == 'service_account':
            # credentials_path may be a path to a file OR a dict-like JSON string
            # If path exists, load from file
            if isinstance(self.credentials_path, str) and os.path.exists(self.credentials_path):
                creds = service_account.Credentials.from_service_account_file(
                    self.credentials_path, scopes=SCOPES
                )
            else:
                # Try to interpret credentials_path as JSON content
                parsed = None
                if isinstance(self.credentials_path, dict):
                    parsed = self.credentials_path
                else:
                    try:
                        parsed = json.loads(self.credentials_path)
                    except Exception:
                        parsed = None

                if parsed:
                    if parsed.get("private_key"):
                        parsed["private_key"] = parsed["private_key"].replace("\\n", "\n")
                    creds = service_account.Credentials.from_service_account_info(parsed, scopes=SCOPES)
                else:
                    raise RuntimeError("Invalid service account credentials for service_account auth method")
        else:
            # Fallback to OAuth flow (interactive)
            if os.path.exists('token.pickle'):
                with open('token.pickle', 'rb') as token:
                    creds = pickle.load(token)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, SCOPES
                    )
                    creds = flow.run_local_server(port=0)

                with open('token.pickle', 'wb') as token:
                    pickle.dump(creds, token)

        return build('drive', 'v3', credentials=creds)

    def get_all_text_files(self, folder_id: str) -> List[Dict[str, str]]:
        # Recursively traverse and return files as dicts with id, name, path, mimeType
        return self._traverse_folder(folder_id, path="")

    def _traverse_folder(self, folder_id: str, path: str = "") -> List[Dict[str, str]]:
        all_files: List[Dict[str, str]] = []
        page_token = None
        query = f"'{folder_id}' in parents and trashed=false"

        while True:
            results = self.service.files().list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType, parents, capabilities)",
                pageToken=page_token,
                pageSize=100,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()

            items = results.get('files', [])
            for item in items:
                current_path = f"{path}/{item['name']}" if path else item['name']
                if item['mimeType'] == 'application/vnd.google-apps.folder':
                    caps = item.get('capabilities', {})
                    if caps.get('canListChildren', True):
                        all_files.extend(self._traverse_folder(item['id'], current_path))
                    else:
                        # Skip inaccessible folder
                        continue
                else:
                    if self._is_text_file(item):
                        all_files.append({
                            'id': item['id'],
                            'name': item['name'],
                            'path': current_path,
                            'mimeType': item.get('mimeType', ''),
                        })

            page_token = results.get('nextPageToken')
            if not page_token:
                break

        return all_files

    def _is_text_file(self, item: Dict[str, Any]) -> bool:
        mime_type = item.get('mimeType', '')
        name = item.get('name', '')
        text_mime_types = [
            'text/plain', 'text/csv', 'text/html', 'text/markdown',
            'application/json', 'application/xml', 'text/x-python', 'application/javascript'
        ]
        text_extensions = ['.txt', '.md', '.csv', '.json', '.xml', '.log', '.py', '.js', '.java']
        return (mime_type in text_mime_types) or any(name.lower().endswith(ext) for ext in text_extensions)

    def download_file(self, file_id: str) -> Optional[str]:
        """Download a file's bytes from Drive and return decoded text (utf-8, ignore errors)."""
        try:
            request = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)

            done = False
            max_retries = 5
            retries = 0
            while not done:
                try:
                    _, done = downloader.next_chunk()
                except Exception as e:
                    retries += 1
                    if retries < max_retries:
                        # small backoff
                        time.sleep(2 ** retries)
                        continue
                    else:
                        print(f"    ✗ Error downloading file {file_id}: {str(e)}")
                        return None

            fh.seek(0)
            try:
                return fh.read().decode('utf-8', errors='ignore')
            except Exception:
                return fh.read().decode('latin-1', errors='ignore')
        except Exception as e:
            print(f"    ✗ Error downloading file {file_id}: {str(e)}")
            return None


@st.cache_resource
def get_drive_extractor() -> GoogleDriveExtractor:
    # First, prefer full JSON provided via environment variable
    try:
        parsed = get_service_account_info()
        if parsed:
            # Pass parsed dict directly to extractor which accepts dict/string
            return GoogleDriveExtractor(auth_method='service_account', credentials_path=parsed)
    except Exception:
        pass

    # Next, prefer a local service-account.json file in repo root if present
    local_path = os.path.join(os.getcwd(), 'service-account.json')
    if os.path.exists(local_path):
        return GoogleDriveExtractor(auth_method='service_account', credentials_path=local_path)

    # Fall back to OAuth interactive flow if no service account found
    return GoogleDriveExtractor(auth_method='oauth', credentials_path='credentials.json')


@st.cache_data
def list_subfolders(parent_id):
    """List subfolders in Google Drive"""
    try:
        service = get_drive_service()
        results = []
        page_token = None

        while True:
            resp = (
                service.files()
                .list(
                    q=f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
                    fields="nextPageToken, files(id, name)",
                    pageToken=page_token,
                )
                .execute()
            )

            results.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken", None)
            if not page_token:
                break

        return results
    except Exception as e:
        st.error(f"Could not list subfolders: {str(e)}")
        return []


@st.cache_data
def list_txt_in_folder(folder_id):
    """List text files in a Google Drive folder"""
    try:
        extractor = get_drive_extractor()
        # Use extractor to get all text files under the folder and return only id/name
        files = extractor.get_all_text_files(folder_id)
        return [{'id': f['id'], 'name': f['name']} for f in files]
    except Exception as e:
        st.error(f"Could not list TXT files in folder: {str(e)}")
        return []


@st.cache_data
def download_txt_as_text(file_id):
    """Download text file from Google Drive"""
    try:
        extractor = get_drive_extractor()
        content = extractor.download_file(file_id)
        return content or ""
    except Exception as e:
        st.error(f"Failed to download text from drive: {str(e)}")
        return ""


def parse_uploaded_file(uploaded_file):
    """Parse uploaded file content"""
    try:
        fname = uploaded_file.name.lower()

        if fname.endswith(".txt"):
            try:
                bytes_content = uploaded_file.read()
                if isinstance(bytes_content, bytes):
                    return bytes_content.decode("utf-8", errors="ignore")
                else:
                    return str(bytes_content)
            except Exception as e:
                st.error(
                    f"Failed to decode TXT file: {str(e)}. Please use UTF-8 encoding."
                )
                return ""


        elif fname.endswith(".docx"):
            try:
                doc = Document(io.BytesIO(uploaded_file.read()))
                return "\n".join([para.text for para in doc.paragraphs])
            except Exception as e:
                st.error(
                    f"Failed to parse DOCX: {str(e)}. File may be corrupt or wrong format."
                )
                return ""

        elif fname.endswith(".pdf"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp.close()
                    text = extract_pdf_text(tmp.name)
                    os.remove(tmp.name)
                    return text
            except Exception as e:
                st.error(
                    f"Failed to parse PDF: {str(e)}. The file may be encrypted or corrupt."
                )
                return ""
        else:
            st.warning("Unsupported file format. Only .txt, .docx, .pdf allowed.")
            return ""
    except Exception as e:
        st.error(f"Could not parse uploaded file: {str(e)}")
        return ""


def chunk_documents(reference_docs: List[str], chunk_size: int) -> List[Dict[str, Any]]:
    """Split documents into chunks"""
    try:
        chunks = []
        for doc_index, doc in enumerate(reference_docs):
            doc = (doc or "").strip()
            if not doc:
                continue
            for i in range(0, len(doc), chunk_size):
                chunk = doc[i : i + chunk_size]
                if chunk.strip():
                    chunks.append(
                        {"text": chunk, "doc_index": doc_index, "chunk_start": i}
                    )
        return chunks
    except Exception as e:
        error_handler("")
        return []


def safe_openai_call(fn: Any, *args, retries: int = 3, **kwargs) -> Any:
    """Safe OpenAI call with retries"""
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2**attempt)
            else:
                error_handler("")
                return None


def get_embeddings_for_chunks(chunks: List[Dict[str, Any]]) -> List[np.ndarray]:
    """Get embeddings for document chunks"""
    try:
        texts = [chunk["text"] for chunk in chunks]
        max_batch = 96
        out = []

        for i in range(0, len(texts), max_batch):
            response = safe_openai_call(
                client.embeddings.create,
                input=texts[i : i + max_batch],
                model="text-embedding-3-small",
            )

            if response is None or not hasattr(response, "data"):
                error_handler("")
                return []

            emb = [np.array(d.embedding) for d in response.data]
            out.extend(emb)

        return out
    except Exception as e:
        error_handler("")
        return []


def embedding_for_query(query):
    """Get embedding for query"""
    try:
        response = safe_openai_call(
            client.embeddings.create, input=[query], model="text-embedding-3-small"
        )

        if response is None or not hasattr(response, "data"):
            st.error("Failed to compute embedding for query via OpenAI.")
            return np.zeros(1536)

        return np.array(response.data[0].embedding)
    except Exception as e:
        st.error(f"Embedding for query failed: {str(e)}")
        return np.zeros(1536)


def retrieve_relevant_chunks(
    reference_docs: List[str], user_query: str, k: int, chunk_size: int
) -> List[str]:
    """Retrieve relevant chunks using semantic similarity"""
    try:
        # Create a hash for caching
        ref_hash = hash(tuple(reference_docs))

        if (
            "rag_ref_docs_hash" not in st.session_state
            or st.session_state.rag_ref_docs_hash != ref_hash
        ):
            st.session_state.rag_chunks = chunk_documents(reference_docs, chunk_size)
            st.session_state.rag_chunks_embeddings = (
                get_embeddings_for_chunks(st.session_state.rag_chunks)
                if st.session_state.rag_chunks
                else []
            )
            st.session_state.rag_ref_docs_hash = ref_hash

        if not st.session_state.rag_chunks:
            return []

        query_emb = embedding_for_query(user_query)
        chunk_embs = st.session_state.rag_chunks_embeddings

        if not chunk_embs:
            st.warning("No embeddings for context. Try reloading reference docs.")
            return []

        # Batch similarity calculation using numpy for speed
        chunk_embs_np = np.stack(chunk_embs)
        query_emb_np = np.array(query_emb)
        sims = np.dot(chunk_embs_np, query_emb_np) / (
            np.linalg.norm(chunk_embs_np, axis=1) * np.linalg.norm(query_emb_np) + 1e-8
        )

        # Get top k chunks
        idxs = np.argsort(sims)[::-1][:k]
        relevant_chunks = [st.session_state.rag_chunks[i]["text"] for i in idxs]

        return relevant_chunks
    except Exception as e:
        error_handler("")
        return []


def assemble_context(
    reference_docs: List[str], user_query: str, k: int, chunk_size: int
) -> str:
    """Assemble context from relevant chunks"""
    try:
        relevant_chunks = retrieve_relevant_chunks(
            reference_docs, user_query, k, chunk_size
        )

        if not relevant_chunks:
            st.warning("No relevant reference documents found for this query.")

        context_block = "\n\n".join(relevant_chunks)
        return context_block
    except Exception as e:
        error_handler("")
        return ""


def run_model(
    context_block: str,
    proposal_block: Optional[str],
    user_query: str,
    model_name: str,
    config: Dict[str, Any],
) -> str:
    """Run the model with context and query"""
    use_proposal = proposal_block and (
        "proposal" in user_query.lower() or "uploaded document" in user_query.lower()
    )

    prompt = (
        f"""
You are an expert internal auditor and financial policy assistant. Using only the content provided (uploaded files, templates, and references), generate a clear, structured, and accurate report. Do not use external knowledge.

1. Tone & Purpose
Keep your tone professional, clear, and informative.
Reports should be understandable to non-experts but grounded in policy.

2. Structure
Use clear sections with bold titles, such as:
Summary: What the document is and what you're evaluating.
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
Do not say "as per guidelines" without specifying the document.
Do not copy large sections of text—summarize instead.

Reference Documents:
{context_block}
"""
        + (f"\nProposal document:\n{proposal_block}\n" if use_proposal else "")
        + f"""
User Question:
{user_query}

If the answer is not found in the provided context, respond: "The answer is not present in the provided references." Otherwise, answer fully, using a friendly, complete, professional and helpful style.
"""
    )

    try:
        input_tokens = count_tokens(prompt, model_name)

        if input_tokens > (config["TOKEN_BUDGET"] - config["MAX_RESPONSE_TOKENS"]):
            # Attempt truncation
            orig_chunks = context_block.split("\n\n")
            while (
                input_tokens > (config["TOKEN_BUDGET"] - config["MAX_RESPONSE_TOKENS"])
                and len(orig_chunks) > 1
            ):
                orig_chunks = orig_chunks[:-1]
                context_block_new = "\n\n".join(orig_chunks)
                prompt = prompt.replace(context_block, context_block_new)
                context_block = context_block_new
                input_tokens = count_tokens(prompt, model_name)

            if input_tokens > (config["TOKEN_BUDGET"] - config["MAX_RESPONSE_TOKENS"]):
                st.warning("Prompt too long. Try selecting fewer reference documents.")
                return "Error: Token limit exceeded."

        response = safe_openai_call(
            client.chat.completions.create,
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert auditor and policy assistant. Your job is to help the user by providing high-quality, easy to understand, fully structured answers using ONLY the context supplied.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=config["MAX_RESPONSE_TOKENS"],
        )

        if not response or not hasattr(response, "choices"):
            error_handler("")
            return "An error occurred in generating the response."

        return response.choices[0].message.content
    except Exception:
        error_handler("")
        return "Model run error."


def make_summary(full_answer, model_name, config):
    """Generate summary of the full answer"""
    summary_prompt = f"""
Summarize the following answer in 2-4 lines for a 'Summary (TL;DR)' box at the end of a report. Focus on only the essential points and avoid repetition. Write in clear plain English.

Answer:
{full_answer}

TL;DR:
"""
    try:
        response = safe_openai_call(
            client.chat.completions.create,
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant who summarizes text for users in short, plain language for non-experts.",
                },
                {"role": "user", "content": summary_prompt},
            ],
            max_tokens=config["SUMMARY_MAX_TOKENS"],
        )

        if not response or not hasattr(response, "choices"):
            return "Could not generate summary."

        return response.choices[0].message.content
    except Exception as e:
        st.warning(f"Could not generate summary: {str(e)}")
        return "Could not generate summary."


# STREAMLIT UI
def main():
    st.set_page_config(page_title="Internal Audit Officer", layout="wide")
    PASSWORD = os.getenv("PASSWORD")

    # Ask for password if not authenticated yet
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("🔐 Authentication Required")

        with st.form("login_form"):
            pwd = st.text_input(
                "Enter password:", type="password", placeholder="Enter your password"
            )
            submit_button = st.form_submit_button("Login")

        if submit_button:
            if pwd == PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Incorrect password")
        st.stop()

    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image("Masthead.png", width="stretch")
        except Exception:
            pass
        st.markdown(
            "<h3 style='text-align: center;'>Assistant Internal Audit Officer (A-IAO)</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align: center;'>An artificial Intelligence based prototype</p>",
            unsafe_allow_html=True,
        )

    # Hide Streamlit styling
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display: none;}
        .st-emotion-cache-13ln4jf {display: none;}
        .st-emotion-cache-ocqkz7 {display: none;}
        footer {visibility: hidden !important;}
        [data-testid="stToolbar"] {visibility: hidden !important;}
        .block-container { padding-top: 1rem !important; }
        .main { padding-top: 0rem !important; }
        </style>
    """
    try:
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    except Exception:
        pass

    # Initialize Google Drive service
    try:
        drive_service = get_drive_service()
    except Exception as e:
        st.error(f"Google Drive authentication/setup failed: {str(e)}")
        st.stop()

    # Subfolder selection
    st.subheader("Select Reference Subfolders")
    try:
        subfolders = list_subfolders(DRIVE_MAIN_FOLDER_ID)
        if not subfolders:
            st.warning("No subfolders found in project.")
            st.stop()

        subfolder_names = [f["name"] for f in subfolders]
        subfolder_map = {f["name"]: f["id"] for f in subfolders}

        selected_subfolders = st.multiselect(
            "Choose one or more subfolders", subfolder_names
        )
    except Exception as e:
        st.error(f"Listing subfolders failed: {str(e)}")
        st.stop()

    # Initialize reference docs in session state
    if "reference_docs" not in st.session_state:
        st.session_state.reference_docs = []

    # Fetch reference documents
    if selected_subfolders:
        if st.button(f"Fetch and combine files from: {', '.join(selected_subfolders)}"):
            docs = []
            with st.spinner("Fetching reference files..."):
                try:
                    for subfolder_name in selected_subfolders:
                        files_in_sub = list_txt_in_folder(subfolder_map[subfolder_name])
                        for file in files_in_sub:
                            raw = download_txt_as_text(file["id"])
                            if raw:
                                docs.append(raw.strip())

                    st.session_state.reference_docs = docs

                    # Reset RAG cache
                    for k in [
                        "rag_chunks",
                        "rag_chunks_embeddings",
                        "rag_ref_docs_hash",
                    ]:
                        if k in st.session_state:
                            del st.session_state[k]

                    st.success(f"Loaded {len(docs)} reference files.")
                except Exception as e:
                    st.error(f"Reference document fetching error: {str(e)}")
    else:
        st.info("Please select at least one subfolder to load reference documents.")

    reference_docs = st.session_state.reference_docs

    # Display reference docs status
    if reference_docs:
        st.info(f"Currently loaded reference documents: {len(reference_docs)}")
    else:
        st.info("No reference documents loaded.")

    # Mode selection
    st.subheader("Select Mode")
    mode = st.radio(
        "Choose mode",
        ["Report/Template Generation", "Query Answering"],
        horizontal=True,
    )

    # Proposal document upload
    proposal_text = ""
    if mode == "Report/Template Generation":
        st.subheader("Upload Proposal (Optional)")
        try:
            prop_file = st.file_uploader(
                "Upload your proposal or template (.txt, .docx, .pdf)",
                type=["txt", "docx", "pdf"],
            )
            if prop_file:
                with st.spinner("Parsing proposal..."):
                    proposal_text = parse_uploaded_file(prop_file)
                    if proposal_text:
                        char_count = len(proposal_text)
                        word_count = max(1, char_count // 5)
                        st.info(
                            f"Uploaded proposal has approximately {word_count} words ({char_count} characters)."
                        )
        except Exception as e:
            st.error(f"Proposal parsing/upload failed: {str(e)}")

    # Quick prompts for report mode
    if mode == "Report/Template Generation":
        st.subheader("Quick Prompts")
        if "quick_prompt" not in st.session_state:
            st.session_state.quick_prompt = None

        quick_col1, quick_col2, quick_col3 = st.columns(3)
        try:
            with quick_col1:
                if st.button("Finance Concurrence"):
                    st.session_state.quick_prompt = PRESET_QUERIES[
                        "Finance Concurrence"
                    ]
            with quick_col2:
                if st.button("Payment Proposal"):
                    st.session_state.quick_prompt = PRESET_QUERIES["Payment Proposal"]
            with quick_col3:
                if st.button("Internal Audit"):
                    st.session_state.quick_prompt = PRESET_QUERIES["Internal Audit"]
        except Exception:
            pass

    # Model selection
    st.subheader("Select Model")
    try:
        model_cols = st.columns(len(MODEL_MAP))
        for i, (label, model) in enumerate(MODEL_MAP.items()):
            if model_cols[i].button(label):
                st.session_state.selected_model = model
                st.session_state.selected_model_label = label
    except Exception:
        st.session_state.selected_model = "gpt-4.1"
        st.session_state.selected_model_label = (
            "Gamma - Large input capacity, detailed tasks"
        )

    # Initialize model selection if not exists
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gpt-4.1"
    if "selected_model_label" not in st.session_state:
        st.session_state.selected_model_label = (
            "Gamma - Large input capacity, detailed tasks"
        )

    st.success(f"Model selected: {st.session_state.selected_model_label}")
    selected_model = st.session_state.selected_model

    # Get model configuration
    config = MODEL_CONFIGS.get(selected_model, MODEL_CONFIGS["gpt-4.1"])

    # Report/Template Generation Mode
    if mode == "Report/Template Generation":
        st.subheader("Report/Template Query")

        if "user_query" not in st.session_state:
            st.session_state.user_query = ""

        try:
            user_query = st.text_area(
                "Enter a query, or select a Quick Prompt above",
                value=st.session_state.get("user_query", ""),
                height=80,
            )
            if user_query != st.session_state.get("user_query", ""):
                st.session_state.user_query = user_query
        except Exception:
            user_query = ""

        submit_custom_query = st.button("Submit")

        # Handle query submission
        used_query = user_query
        if st.session_state.get("quick_prompt"):
            used_query = st.session_state.quick_prompt

        if (submit_custom_query or st.session_state.get("quick_prompt")) and used_query:
            if reference_docs:
                # Limit proposal text size
                if proposal_text and len(proposal_text) > config["PROPOSAL_CHAR_LIMIT"]:
                    st.warning(
                        f"Proposal is large ({len(proposal_text)} chars). "
                        f"Only the first {config['PROPOSAL_CHAR_LIMIT']} characters will be used."
                    )
                    proposal_text = proposal_text[: config["PROPOSAL_CHAR_LIMIT"]]

                with st.spinner("Fetching relevant context and generating report..."):
                    try:
                        context_block = assemble_context(
                            reference_docs,
                            used_query,
                            config["CONTEXT_CHUNKS"],
                            config["CHUNK_CHAR_LIMIT"],
                        )
                        output = run_model(
                            context_block,
                            proposal_text,
                            used_query,
                            selected_model,
                            config,
                        )
                        summary = make_summary(output, selected_model, config)
                    except Exception as e:
                        st.error(f"Report generation error: {str(e)}")
                        output = "Error generating report"
                        summary = "Error"

                # Display results
                st.subheader("Result")
                try:
                    st.write(output)
                    st.markdown(
                        f"---\n<b>Summary (TL;DR):</b><br>{summary}",
                        unsafe_allow_html=True,
                    )

                    # Download button
                    download_content = output + "\n\nSummary (TL;DR):\n" + summary
                    st.download_button(
                        "Download response as TXT",
                        download_content,
                        file_name="audit_response.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"Error displaying results: {str(e)}")

                # Reset quick prompt
                st.session_state.quick_prompt = None
            else:
                st.info("Please select and load reference documents from Google Drive.")

    # Query Answering Mode
    elif mode == "Query Answering":
        st.subheader("Ask a Question (No proposal necessary)")

        if "user_query_qa" not in st.session_state:
            st.session_state.user_query_qa = ""

        try:
            user_query_qa = st.text_area(
                "Ask a policy/guideline question",
                value=st.session_state.get("user_query_qa", ""),
                height=80,
            )
            if user_query_qa != st.session_state.get("user_query_qa", ""):
                st.session_state.user_query_qa = user_query_qa

            submit_query_qa = st.button("Get Answer")
        except Exception:
            user_query_qa = ""
            submit_query_qa = False

        if submit_query_qa and user_query_qa:
            if reference_docs:
                with st.spinner("Searching references and answering..."):
                    try:
                        context_block = assemble_context(
                            reference_docs,
                            user_query_qa,
                            config["CONTEXT_CHUNKS"],
                            config["CHUNK_CHAR_LIMIT"],
                        )
                        output = run_model(
                            context_block, None, user_query_qa, selected_model, config
                        )
                        summary = make_summary(output, selected_model, config)
                    except Exception as e:
                        st.error(f"QA error: {str(e)}")
                        output = "Error generating answer"
                        summary = "Error"

                # Display results
                st.subheader("Answer")
                try:
                    st.write(output)
                    st.markdown(
                        f"---\n<b>Summary (TL;DR):</b><br>{summary}",
                        unsafe_allow_html=True,
                    )

                    # Download button
                    download_content = output + "\n\nSummary (TL;DR):\n" + summary
                    st.download_button(
                        "Download answer as TXT",
                        download_content,
                        file_name="query_answer.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"Error displaying answer: {str(e)}")
            else:
                st.info("Please select and load reference documents from Google Drive.")


if __name__ == "__main__":
    main()

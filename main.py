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

CONTEXT_CHUNKS = 40
CHUNK_CHAR_LIMIT = 2000
PROPOSAL_CHAR_LIMIT = 30000
TOKEN_BUDGET = 125000
MAX_RESPONSE_TOKENS = 2500
SUMMARY_MAX_TOKENS = 512

try:
    client = OpenAI(api_key=API_KEY)
except Exception:
    st.error("An internal error occurred. Please report error 1.")  # ERROR 1 - Check OpenAI Key
    st.stop()


def count_tokens(text, model="gpt-4o"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


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
    except Exception:
        st.error("An internal error occurred. Please report error 2.")  # ERROR 2 - Google Drive
        st.stop()


def list_subfolders(service, parent_id):
    try:
        results = service.files().list(
            q=f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
            fields="files(id, name)").execute()
        return results.get('files', [])
    except Exception:
        st.error("An internal error occurred. Please report error 3.")  # ERROR 3 - Reference Docs
        return []


def list_txt_in_folder(service, folder_id):
    try:
        results = service.files().list(
            q=f"'{folder_id}' in parents and mimeType='text/plain' and trashed=false",
            pageSize=500,
            fields="files(id, name)").execute()
        return results.get('files', [])
    except Exception:
        st.error("An internal error occurred. Please report error 4.")  # ERROR 4
        return []


def download_txt_as_text(service, file_id):
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        return fh.read().decode('utf-8', errors='ignore')
    except Exception:
        st.error("An internal error occurred. Please report error 5.")  # ERROR 5
        return ""


def parse_uploaded_file(uploaded_file):
    fname = uploaded_file.name.lower()
    try:
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
    except Exception:
        st.error("An internal error occurred. Please report error 6.")  # ERROR 6
    return ""


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
    try:
        for i in range(0, len(texts), max_batch):
            response = client.embeddings.create(
                input=texts[i:i+max_batch],
                model="text-embedding-3-small"
            )
            emb = [np.array(d.embedding) for d in response.data]
            out.extend(emb)
    except Exception:
        st.error("An internal error occurred. Please report error 7.")  # ERROR 7
    return out


def embedding_for_query(query):
    try:
        response = client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        )
        return np.array(response.data[0].embedding)
    except Exception:
        st.error("An internal error occurred. Please report error 8.")  # ERROR 8
        return np.zeros(1536)


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
    try:
        relevant_chunks = retrieve_relevant_chunks(reference_docs, user_query, k=k)
        return "\n\n".join(relevant_chunks)
    except Exception:
        st.error("An internal error occurred. Please report error 9.")  # ERROR 9
        return ""


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
    except Exception:
        st.error("An internal error occurred. Please report error 10.")  # ERROR 10
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
        st.error("An internal error occurred. Please report error 11.")  # ERROR 11
        return "Could not generate summary."

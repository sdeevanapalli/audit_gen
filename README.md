
# Internal Audit Officer (IAO)

An advanced, scalable AI-powered audit automation platform architected on a Retrieval-Augmented Generation (RAG) paradigm, leveraging state-of-the-art Large Language Models (LLMs) via OpenAI API to facilitate high-precision internal audit documentation, compliance verification, and financial concurrence synthesis.

## Core Functionalities

- **LLM-Driven Generative Intelligence:** Utilizes OpenAI’s GPT-3.5, GPT-4, and GPT-4o models to dynamically generate nuanced, contextually-grounded audit narratives, payment proposals, and compliance reports conforming to regulatory frameworks.
- **Retrieval-Augmented Generation (RAG) Framework:** Implements an efficient document retrieval subsystem interfacing with Google Drive’s API, ingesting and semantically chunking voluminous regulatory reference corpora to enable context-aware prompt conditioning for the LLMs.
- **Multi-Modal Document Parsing:** Robust parsing pipeline supporting plaintext (`.txt`), Microsoft Word (`.docx`), and Portable Document Format (`.pdf`), employing `python-docx` and `pdfminer.six` libraries for high-fidelity text extraction under variable document structures.
- **Token-Efficient Prompt Engineering:** Employs `tiktoken` tokenization algorithms for rigorous budget enforcement, intelligently truncating and prioritizing reference contexts to optimize LLM input tokens while mitigating truncation-induced semantic loss.
- **Interactive Streamlit-Based UX:** Provides a modular, reactive UI featuring:
  - One-click Quick Prompt invocations aligned with common audit workflows.
  - Dynamic selection among heterogeneous LLM variants balancing latency and output granularity.
  - Asynchronous file upload mechanisms with real-time content parsing.
  - Downloadable output persistence facilitating audit trail maintenance.

## Technology Stack

- **Frontend & Orchestration:** Streamlit for rapid prototyping and deployment of interactive AI-driven interfaces.
- **LLM Backend:** OpenAI GPT-3.5, GPT-4, and GPT-4o models accessed via OpenAI Python SDK.
- **Retrieval Layer:** Google Drive REST API integration, authenticated through OAuth2 service accounts, implementing recursive folder enumeration and MIME-type filtering for document retrieval.
- **Document Processing:** `python-docx` for DOCX parsing, `pdfminer.six` for PDF content extraction, and native UTF-8 decoding pipelines.
- **Token Management:** OpenAI’s `tiktoken` library for precise token accounting and input size constraint adherence.
- **Environment & Configuration:** Managed via `.env` files utilizing `python-dotenv` to secure API credentials and service account JSON payloads.


## Operational Considerations

* The system orchestrates token-budgeted prompt concatenation, dynamically truncating reference document chunks to comply with OpenAI’s input constraints.
* Selection between GPT model variants affords fine-grained control over inference latency versus output detail and verbosity.
* All API interactions adhere to OAuth2 best practices with ephemeral credential handling to ensure security compliance.
* Designed for extensibility to incorporate additional document types and downstream audit automation workflows.

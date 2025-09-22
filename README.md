# Internal Audit Officer (IAO)

The Internal Audit Officer (IAO) is an AI-powered platform built to make audit automation smarter, faster, and more reliable. It’s designed on a Retrieval-Augmented Generation (RAG) framework and uses advanced large language models to handle everything from audit documentation to compliance checks and financial reporting with high accuracy.

## Core Features

- **Intelligent Audit Assistance**  
  IAO uses OpenAI’s GPT models (GPT-3.5, GPT-4, GPT-4o) to generate well-structured audit narratives, compliance reports, and payment proposals. All outputs are aligned with standard regulatory and financial practices.  

- **Context-Aware Retrieval**  
  The system connects to Google Drive, retrieves documents, and breaks them into smaller, meaningful sections. This ensures that the LLM works with precise context when creating reports or verifying compliance.  

- **Flexible Document Handling**  
  It can process multiple file types including TXT, DOCX, and PDF. Tools like `python-docx` and `pdfminer.six` are used to extract content accurately, even from complex document layouts.  

- **Smart Token Management**  
  To work within LLM input limits, the platform uses `tiktoken` to manage tokens efficiently. Important sections of text are prioritized so that key details are never lost.  

- **Streamlined User Experience**  
  The Streamlit-based interface makes it easy for users to interact with the system. Features include:  
  - Quick prompts for common audit tasks  
  - Options to choose between different LLM models depending on speed and detail requirements  
  - Drag-and-drop file uploads with instant parsing  
  - One-click downloads to save generated reports and maintain audit trails  

## Technology Stack

- **Frontend & Orchestration:** Streamlit  
- **LLM Backend:** OpenAI GPT-3.5, GPT-4, GPT-4o via the OpenAI Python SDK  
- **Retrieval Layer:** Google Drive REST API with OAuth2 authentication  
- **Document Processing:** `python-docx`, `pdfminer.six`, UTF-8 text pipelines  
- **Token Management:** OpenAI `tiktoken`  
- **Configuration:** Environment variables managed with `python-dotenv`  

## How It Works in Practice

The platform dynamically adjusts how much context to include in each LLM request, always staying within token limits. Users can decide which GPT model to use depending on whether speed or detail is the priority. Security is maintained through proper credential handling and OAuth2 standards.  

IAO is built with flexibility in mind. It can be extended to support new document types and adapted for more specialized audit workflows as organizations grow.  

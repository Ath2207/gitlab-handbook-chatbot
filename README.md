# ?? GitLab Handbook AI Assistant

A RAG-based chatbot that answers questions grounded in GitLab's public Handbook and Direction pages.

## How it works

- Fetches 11 GitLab Handbook pages
- Splits into chunks using RecursiveCharacterTextSplitter
- Embeds locally using HuggingFace all-MiniLM-L6-v2
- Stores in ChromaDB persisted to disk
- Retrieves with MMR search
- Answers using Gemini 2.5 Flash
- Displays in Streamlit chat UI with source citations

## Setup

### 1. Prerequisites
- Python 3.10+
- Google AI API key from https://aistudio.google.com/

### 2. Clone the repo
git clone https://github.com/YOUR_USERNAME/gitlab-handbook-chatbot.git

### 3. Create virtual environment
python -m venv venv
venv\Scripts\activate

### 4. Install dependencies
pip install -r requirements.txt

### 5. Set up API key
Create a .env file:
GOOGLE_API_KEY=your-gemini-api-key-here

### 6. Run the app
streamlit run app.py

## First Run
On first run the app fetches and indexes all handbook pages into chroma_db folder. Subsequent runs load from disk instantly.

## Key Design Decisions
- RAG: Prevents hallucinations, answers grounded in handbook
- HuggingFace Embeddings: Runs locally, no API quota
- ChromaDB with persistence: No re-indexing on restart
- MMR retrieval: Diverse and relevant chunks
- Source citations: Transparency for every answer
- Gemini 2.5 Flash: Fast, free tier via Google AI Studio

## Troubleshooting
ModuleNotFoundError langchain_chroma: pip install langchain-chroma
ModuleNotFoundError torchvision: pip install torchvision
Quota errors: Only Gemini chat calls use quota, embeddings are local

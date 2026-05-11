from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate


PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "gitlab_handbook"

SOURCE_URLS = [
    "https://handbook.gitlab.com/handbook/values/",
    "https://handbook.gitlab.com/handbook/communication/",
    "https://handbook.gitlab.com/handbook/people-group/",
    "https://handbook.gitlab.com/handbook/engineering/",
    "https://handbook.gitlab.com/handbook/product/",
    "https://handbook.gitlab.com/handbook/marketing/",
    "https://handbook.gitlab.com/handbook/finance/",
    "https://handbook.gitlab.com/handbook/legal/",
    "https://handbook.gitlab.com/handbook/security/",
    "https://handbook.gitlab.com/handbook/sales/",
    "https://about.gitlab.com/direction/",
]

SYSTEM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert assistant for GitLab's Handbook and Direction pages.
Answer using ONLY the context provided. If the context doesn't have enough information, say so clearly.
At the end of your answer, mention which topic area your answer draws from.

Context:
{context}

Question: {question}

Answer:""",
)


st.set_page_config(page_title="GitLab Handbook Assistant", page_icon="🦊", layout="wide")


with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_key = st.text_input("Google AI API Key", type="password",
                            help="Get a free key at https://aistudio.google.com/")
    st.markdown("---")
    st.info(
        "This assistant uses **RAG** (Retrieval-Augmented Generation).\n\n"
        "It fetches content from GitLab's Handbook and answers your questions "
        "grounded in that content — not general AI knowledge."
    )
    st.markdown("### 📄 Sources indexed")
    for url in SOURCE_URLS:
        label = url.rstrip("/").split("/")[-1].replace("-", " ").title()
        st.markdown(f"- [{label}]({url})")
    st.markdown("---")
    rebuild = st.button("🔄 Re-index Handbook", use_container_width=True)


st.title("🦊 GitLab Handbook Assistant")
st.caption("Ask anything about GitLab's values, processes, engineering practices, or product direction.")


if not api_key:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GOOGLE_API_KEY"]
        except:
            pass

if not api_key:
    st.warning("Enter your Google AI API Key in the sidebar to begin.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key


def build_vector_store(force_rebuild=False):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if not force_rebuild and os.path.exists(PERSIST_DIR):
        st.toast("✅ Loaded existing index from disk.")
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )

    status = st.status("📥 Fetching GitLab Handbook pages…", expanded=True)
    all_docs = []

    for url in SOURCE_URLS:
        try:
            status.write(f"Loading: {url}")
            loader = WebBaseLoader(web_paths=[url])
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_url"] = url
            all_docs.extend(docs)
        except Exception as e:
            status.write(f"⚠️ Failed: {url} — {e}")

    if not all_docs:
        status.update(label="❌ No pages could be loaded.", state="error")
        st.error("Could not load any handbook pages. Check your network and try again.")
        st.stop()

    status.write(f"✂️ Splitting {len(all_docs)} pages into chunks…")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)
    status.write(f"   → {len(chunks)} chunks created.")

    status.write("🔢 Embedding and building vector store…")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    status.update(label=f"✅ Indexed {len(chunks)} chunks from {len(all_docs)} pages.", state="complete")
    return vectorstore


if rebuild and "vectorstore" in st.session_state:
    del st.session_state["vectorstore"]

if "vectorstore" not in st.session_state:
    try:
        st.session_state["vectorstore"] = build_vector_store(force_rebuild=rebuild)
    except Exception as e:
        st.error(f"Failed to build vector store: {e}")
        st.stop()

vectorstore = st.session_state["vectorstore"]


@st.cache_resource
def get_qa_chain(_vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        convert_system_message_to_human=True,
    )
    retriever = _vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6},
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": SYSTEM_PROMPT},
    )

try:
    qa_chain = get_qa_chain(vectorstore)
except Exception as e:
    st.error(f"Could not initialise QA chain: {e}")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources used"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

if prompt := st.chat_input("Ask about GitLab's values, processes, or product strategy…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the handbook…"):
            try:
                result = qa_chain.invoke({"query": prompt})
                answer = result["result"]
                raw_sources = [
                    doc.metadata.get("source_url") or doc.metadata.get("source", "Unknown")
                    for doc in result.get("source_documents", [])
                ]
                unique_sources = list(dict.fromkeys(raw_sources))

                st.markdown(answer)
                if unique_sources:
                    with st.expander("📄 Sources used"):
                        for src in unique_sources:
                            st.markdown(f"- {src}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": unique_sources,
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")
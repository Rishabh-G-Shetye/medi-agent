import streamlit as st
import os
import tempfile
import re
from src.rag_engine import ClinicalKnowledgeBase
from src.llm_client import GeminiClient

# --- Page Config ---
st.set_page_config(
    page_title="Medi-Agent Pro",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="expanded"
)


# --- 0. Authentication System ---
def check_password():
    """Simple password protection for the application."""

    def password_entered():
        if st.session_state["password"] == "Clinical2025":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter Access Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Enter Access Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True


if not check_password():
    st.stop()

# --- Main Application ---
st.title("🏥 Medi-Agent: Clinical Guideline Assistant")
st.markdown("---")

# --- 1. Session State Initialization ---
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = ClinicalKnowledgeBase()
if "llm_client" not in st.session_state:
    st.session_state.llm_client = GeminiClient()
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. Sidebar: Controls ---
with st.sidebar:
    st.header("🗂️ Knowledge Base")

    # A. Persistence Control
    st.subheader("Storage")
    if st.button("📂 Load Saved Database", use_container_width=True):
        if st.session_state.rag_engine.load_index():
            st.success("Loaded saved guidelines!")
        else:
            st.warning("No database found.")

    st.divider()

    # B. Document Upload
    st.subheader("Ingestion")
    uploaded_files = st.file_uploader("Upload Guidelines (PDF)", type="pdf", accept_multiple_files=True)

    if uploaded_files and st.button("⚡ Build & Save", use_container_width=True):
        with st.spinner("Processing & Indexing..."):
            temp_paths = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
                    tmp.write(uploaded_file.read())
                    temp_paths.append(tmp.name)

            # Process & Save
            st.session_state.rag_engine.load_and_process_pdfs(temp_paths)
            st.session_state.rag_engine.save_index()

            st.success("Database Updated Successfully!")

            # Cleanup
            for path in temp_paths:
                try:
                    os.remove(path)
                except:
                    pass


# --- Helper: UI Components ---
def display_source_cards(context_text):
    """Renders interactive source citations."""
    pattern = r"\[Source: '(.*?)', Page: (\d+)\]"
    parts = re.split(pattern, context_text)

    if len(parts) < 2: return

    for i in range(1, len(parts), 3):
        if i + 2 < len(parts):
            filename = parts[i]
            page_num = parts[i + 1]
            content = parts[i + 2].strip()

            with st.expander(f"📄 Evidence: {filename} (Page {page_num})"):
                st.markdown("**Context Used:**")
                st.caption(content)


# --- 3. Chat Interface ---
# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle Input
if prompt := st.chat_input("Ask a clinical question (e.g., 'Treatment for P. vivax?')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.rag_engine.index:
            st.warning("⚠️ Knowledge base is empty. Please upload PDFs or load a saved database.")
            response = "⚠️ Please upload a PDF or Load a Saved Database first."
        else:
            with st.spinner("Thinking..."):
                context = st.session_state.rag_engine.search(prompt)

                if not context:
                    response = "No relevant information found in the documents."
                else:
                    response = st.session_state.llm_client.generate_response(
                        context=context,
                        query=prompt,
                        chat_history=st.session_state.messages[-5:]
                    )

                st.markdown(response)

                if context:
                    st.divider()
                    st.markdown("### 🔍 Source Verification")
                    display_source_cards(context)

    st.session_state.messages.append({"role": "assistant", "content": response})
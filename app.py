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
def display_source_chips(context_text):
    """
    Renders sources as small, hoverable 'chips' using HTML/CSS.
    (Fixed: Removes indentation to prevent HTML being rendered as code blocks)
    """

    # CSS for the hoverable chips
    tooltip_css = """
    <style>
    .source-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
    }
    .source-chip {
        position: relative;
        display: inline-flex;
        align-items: center;
        background-color: #f0f2f6;
        border: 1px solid #d0d2d6;
        color: #31333F;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        cursor: help;
        transition: background 0.2s;
    }
    /* Dark mode adjustment */
    @media (prefers-color-scheme: dark) {
        .source-chip {
            background-color: #262730;
            border: 1px solid #464b5d;
            color: #fafafa;
        }
    }
    .source-chip:hover {
        background-color: #e0e2e6;
    }
    @media (prefers-color-scheme: dark) {
        .source-chip:hover {
            background-color: #363945;
        }
    }

    /* Tooltip Text */
    .source-chip .tooltip-text {
        visibility: hidden;
        width: 300px;
        background-color: #0e1117;
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 100;
        bottom: 135%;
        left: 50%;
        margin-left: -150px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.8rem;
        line-height: 1.4;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.5);
        border: 1px solid #30333f;
        white-space: normal;
    }
    .source-chip:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """
    st.markdown(tooltip_css, unsafe_allow_html=True)

    # Regex to extract filename, page, AND the content following it
    pattern = r"\[Source: '(.*?)', Page: (\d+)\]"
    parts = re.split(pattern, context_text)

    unique_sources = {}

    if len(parts) > 1:
        for i in range(1, len(parts), 3):
            if i + 1 < len(parts):
                filename = parts[i]
                page = parts[i + 1]
                content_raw = parts[i + 2] if i + 2 < len(parts) else ""

                # Clean content for tooltip
                clean_content = content_raw.strip()[:400] + "..."
                # Escape quotes to prevent HTML breakage
                clean_content = clean_content.replace('"', '&quot;').replace("'", "&#39;")

                identifier = f"{filename} (p.{page})"
                if identifier not in unique_sources:
                    unique_sources[identifier] = clean_content

    if unique_sources:
        # --- HTML CONSTRUCTION (Compact to prevent indentation errors) ---
        html_code = '<div class="source-container">'

        for i, (source_id, content) in enumerate(unique_sources.items()):
            if i >= 3: break
            # IMPORTANT: This f-string is now one line to prevent Markdown code-blocking
            html_code += f'<div class="source-chip">📄 {source_id}<span class="tooltip-text"><b>Context:</b><br>{content}</span></div>'

        html_code += "</div>"

        st.caption("🔍 Sources (Hover for details):")
        st.markdown(html_code, unsafe_allow_html=True)

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
                # Search the vector database
                context = st.session_state.rag_engine.search(prompt)

                if not context:
                    response = "No relevant information found in the documents."
                else:
                    # Generate Answer
                    response = st.session_state.llm_client.generate_response(
                        context=context,
                        query=prompt,
                        chat_history=st.session_state.messages[-5:]
                    )

                st.markdown(response)

                # Display Cleaned Source Chips (UPDATED CALL)
                if context:
                    st.divider()
                    display_source_chips(context)

    st.session_state.messages.append({"role": "assistant", "content": response})
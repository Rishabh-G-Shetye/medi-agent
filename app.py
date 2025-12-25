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
    This prevents clutter while keeping the evidence accessible.
    """

    # CSS for the hoverable chips (Dark/Light mode compatible)
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
        background-color: rgba(128, 128, 128, 0.1);
        border: 1px solid rgba(128, 128, 128, 0.2);
        color: inherit;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        cursor: help;
        transition: background 0.2s;
    }
    .source-chip:hover {
        background-color: rgba(128, 128, 128, 0.2);
    }
    /* Tooltip Text */
    .source-chip .tooltip-text {
        visibility: hidden;
        width: 350px;
        background-color: #262730;
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 100;
        bottom: 125%; 
        left: 50%;
        margin-left: -175px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.85rem;
        line-height: 1.4;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #444;
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

    # Use a dictionary to keep unique sources
    unique_sources = {}

    # parts[0] is usually empty or intro text.
    # The regex split creates groups of 3: [text_before, filename, page, text_after...]
    if len(parts) > 1:
        for i in range(1, len(parts), 3):
            if i + 1 < len(parts):
                filename = parts[i]
                page = parts[i + 1]
                # The content is usually in the next part (parts[i+2])
                content_raw = parts[i + 2] if i + 2 < len(parts) else ""

                # Clean content: limit to 300 chars for the tooltip & escape HTML
                clean_content = content_raw.strip()[:400] + "..."
                clean_content = clean_content.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')

                identifier = f"{filename} (p.{page})"
                if identifier not in unique_sources:
                    unique_sources[identifier] = clean_content

    # Render HTML Chips
    if unique_sources:
        html_code = '<div class="source-container">'

        # Limit to top 3 to keep it clean
        for i, (source_id, content) in enumerate(unique_sources.items()):
            if i >= 3: break

            html_code += f"""
            <div class="source-chip">
                📄 {source_id}
                <span class="tooltip-text"><b>Context:</b><br>{content}</span>
            </div>
            """

        html_code += "</div>"

        # Display "Sources" label (subtle)
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
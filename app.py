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
    """Simple password protection."""

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
    if st.button("📂 Load Saved Database", use_container_width=True):
        if st.session_state.rag_engine.load_index():
            st.success("Loaded saved guidelines!")
        else:
            st.warning("No database found.")

    st.divider()

    # B. Mode Selection
    st.subheader("⚙️ Settings")
    patient_mode = st.toggle("Patient-Friendly Mode", value=False,
                             help="Switch to simple language for explaining to patients.")

    st.divider()

    # C. Document Upload
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
            for path in temp_paths:
                try:
                    os.remove(path)
                except:
                    pass


# --- Helper: UI Components ---
def display_source_chips(context_text):
    """Renders sources as small, hoverable 'chips' using HTML/CSS."""
    tooltip_css = """
    <style>
    .source-container { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    .source-chip {
        position: relative; display: inline-flex; align-items: center;
        background-color: #f0f2f6; border: 1px solid #d0d2d6; color: #31333F;
        padding: 4px 12px; border-radius: 16px; font-size: 0.85rem;
        cursor: help; transition: background 0.2s;
    }
    .source-chip:hover { background-color: #e0e2e6; }
    /* Dark mode */
    @media (prefers-color-scheme: dark) {
        .source-chip { background-color: #262730; border: 1px solid #464b5d; color: #fafafa; }
        .source-chip:hover { background-color: #363945; }
    }
    /* Tooltip */
    .source-chip .tooltip-text {
        visibility: hidden; width: 300px; background-color: #0e1117; color: #fff;
        text-align: left; border-radius: 6px; padding: 10px; position: absolute;
        z-index: 100; bottom: 135%; left: 50%; margin-left: -150px; opacity: 0;
        transition: opacity 0.3s; font-size: 0.8rem; line-height: 1.4;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.5); border: 1px solid #30333f; white-space: normal;
    }
    .source-chip:hover .tooltip-text { visibility: visible; opacity: 1; }
    </style>
    """
    st.markdown(tooltip_css, unsafe_allow_html=True)
    pattern = r"\[Source: '(.*?)', Page: (\d+)\]"
    parts = re.split(pattern, context_text)
    unique_sources = {}

    if len(parts) > 1:
        for i in range(1, len(parts), 3):
            if i + 1 < len(parts):
                filename, page = parts[i], parts[i + 1]
                content_raw = parts[i + 2] if i + 2 < len(parts) else ""
                clean_content = content_raw.strip()[:400].replace('"', '&quot;').replace("'", "&#39;") + "..."
                identifier = f"{filename} (p.{page})"
                if identifier not in unique_sources: unique_sources[identifier] = clean_content

    if unique_sources:
        html_code = '<div class="source-container">'
        for i, (sid, content) in enumerate(unique_sources.items()):
            if i >= 3: break
            html_code += f'<div class="source-chip">📄 {sid}<span class="tooltip-text"><b>Context:</b><br>{content}</span></div>'
        html_code += "</div>"
        st.caption("🔍 Sources (Hover for details):")
        st.markdown(html_code, unsafe_allow_html=True)


# --- 3. Chat Interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a clinical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # --- 0. FAST TRACK (Casual Chat) ---
        if st.session_state.llm_client.check_is_casual(prompt):
            with st.spinner("..."):
                response = st.session_state.llm_client.generate_lightweight_response(prompt)
            st.markdown(response)

        # --- 1. MEDICAL TRACK (RAG) ---
        else:
            if not st.session_state.rag_engine.index:
                st.warning("⚠️ Database empty. Load or Build first.")
                response = "Please load a database."
            else:
                with st.status("🤖 Orchestrating Agents...", expanded=True) as status:
                    st.write("📚 **Retrieval Engine:** Searching knowledge base...")
                    context = st.session_state.rag_engine.search(prompt)

                    if not context:
                        status.update(label="❌ No info found", state="error")
                        response = "No relevant information found in the documents."
                        raw_response_for_chips = ""
                    else:
                        st.write("✅ **Retrieval:** Found relevant guidelines.")

                        # Multi-Agent Pipeline
                        raw_response_for_chips = st.session_state.llm_client.orchestrate_response(
                            context=context,
                            query=prompt,
                            chat_history=st.session_state.messages[-5:],
                            is_patient_mode=patient_mode,
                            status_callback=st.write
                        )

                        # Clean Response (Remove citations from display text)
                        response = re.sub(r"\[Source: '(.*?)', Page: (\d+)\]", "", raw_response_for_chips).strip()

                        status.update(label="✅ Response Generated", state="complete")

                st.markdown(response)

                # Show Chips
                if context and 'raw_response_for_chips' in locals() and raw_response_for_chips:
                    st.divider()
                    display_source_chips(raw_response_for_chips)

    st.session_state.messages.append({"role": "assistant", "content": response})
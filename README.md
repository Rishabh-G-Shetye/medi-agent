# 🏥 Medi-Agent: Clinical Guideline Assistant (RAG)

Medi-Agent is a local, secure, and offline-capable RAG (Retrieval-Augmented Generation) application designed to assist medical professionals in querying complex clinical guidelines.

Built with **Streamlit**, **Google Gemini**, and **FAISS**.

## 🚀 Features

* **Clinical RAG Engine:** Accurate retrieval using sliding-window chunking (no data loss).
* **Smart Citations:** Every answer cites the specific **Source Filename** and **Page Number**.
* **Source Verification:** Interactive UI to view the exact text snippet used by the AI.
* **Persistence:** Save/Load the vector database to disk (no need to re-index every time).
* **Security:** Password protection and prompt injection guardrails.

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/medi-agent.git](https://github.com/YOUR_USERNAME/medi-agent.git)
    cd medi-agent
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Key**
    * Open `src/config.py` and add your Google Gemini API Key.
    * *(Note: In production, use environment variables).*

## 🏥 Usage

1.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

2.  **Login**
    * Default Password: `Clinical2025` (Change this in `app.py`)

3.  **Workflow**
    * Upload a Clinical Guideline PDF (e.g., Malaria Treatment Guidelines).
    * Click **"⚡ Build & Save Knowledge Base"**.
    * Ask questions like *"What is the dosage for P. vivax?"*.
    * Verify the answer using the **Source Verification** dropdowns.

## ⚠️ Disclaimer
This tool is for **informational and research purposes only**. It does not replace professional medical judgment. Always verify AI outputs against the original source text provided in the interface.
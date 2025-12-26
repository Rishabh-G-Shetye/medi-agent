# 🏥 Medi-Agent Pro v2.0

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![AI Model](https://img.shields.io/badge/AI-Gemini%203%20Flash%20%7C%20Gemma%203-4285F4)
![License](https://img.shields.io/badge/License-MIT-green)

> **⚠️ IMPORTANT MEDICAL WARNING**
>
> **Medi-Agent Pro is a Clinical Decision Support System (CDSS) prototype intended for educational and assistance purposes only.**
> * It does **NOT** replace professional medical judgment.
> * **NEVER** use this tool for emergency triage or life-critical decisions.
> * All outputs (dosages, protocols, diagnoses) **MUST** be verified against official medical guidelines and standard sources before clinical application.

---

**Medi-Agent Pro** is a state-of-the-art clinical assistant powered by Google's newest **Gemini 3** reasoning models. It orchestrates a multi-agent workflow to analyze complex medical guidelines and translate them into precise protocols for doctors or empathetic explanations for patients.

---

## 🚀 Key Features (v2.0 Update)

### 🧠 **1. Multi-Agent Reasoning Architecture**
Instead of a single LLM call, Medi-Agent uses a specialized agentic pipeline:
* **🕵️ Researcher Agent (`gemini-3-flash-preview`):** Scans documents for raw facts, contraindications, and dosages. It is purely objective and citation-focused.
* **✍️ Writer Agent (`gemini-3-flash-preview`):** Synthesizes the researcher's notes into the final output. It adapts its tone based on the user's role (Doctor vs. Patient).

### ⚡ **2. Zero-Latency "Fast Track"**
* **Casual Chat:** The system instantly detects greetings (e.g., *"Hello", "How are you"*) and routes them to **`gemma-3-27b-it`** for immediate, low-latency responses, bypassing the heavy RAG pipeline.

### ❤️ **3. Patient-Friendly Mode**
* **One-Click Toggle:** Switch between **Clinician Mode** (Executive summaries, bullet points) and **Patient Mode** (5th-grade reading level, empathetic, "What this means for you").
* **Context Aware:** The AI explains complex terms like *P. falciparum* or *HbA1c* simply when in Patient Mode.

### 🔍 **4. Reasoning Dashboard**
* **Real-Time Visualization:** Watch the agents work in real-time. The UI shows the status of the Retrieval Engine, Researcher Agent, and Writer Agent as they process your query.
* **Smart Citations:** Hoverable "Source Chips" allow you to verify every claim against the original PDF without cluttering the text.

---

## 🛠️ System Architecture

```mermaid
graph TD
    User[User Input] --> Router{Is it Medical?}
    
    %% Fast Track
    Router -- No (Greeting) --> Gemma[⚡ Gemma 3-27B]
    Gemma --> Response
    
    %% Heavy Track
    Router -- Yes (Medical) --> RAG[📚 RAG Retrieval]
    RAG --> Context
    
    subgraph "Gemini 3 Reasoning Engine"
        Context --> Researcher[🕵️ Researcher Agent]
        Researcher -->|Raw Facts + Citations| Writer[✍️ Writer Agent]
        
        Writer -->|Format: Clinician| Output1[Doctor Summary]
        Writer -->|Format: Patient| Output2[Empathetic Explanation]
    end
    
    Output1 --> Response
    Output2 --> Response
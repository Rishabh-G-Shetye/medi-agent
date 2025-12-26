from google import genai
from src.config import Config


class GeminiClient:
    def __init__(self):
        # Initialize the new SDK Client
        self.client = genai.Client(api_key=Config.API_KEY)

    def check_is_casual(self, query: str) -> bool:
        """
        Determines if a query is just casual chatter to skip the heavy RAG pipeline.
        Returns True if it's a greeting/small talk, False if it looks medical.
        """
        clean_q = ''.join(e for e in query.lower() if e.isalnum() or e.isspace()).strip()

        # Exact matches for common greetings
        casual_triggers = {
            "hello", "hi", "hey", "greetings", "good morning", "good afternoon",
            "good evening", "hi there", "hello there", "how are you", "how are you doing",
            "thanks", "thank you", "cool", "ok", "okay", "bye", "goodbye", "help"
        }

        if clean_q in casual_triggers:
            return True

        # Check for short greetings (< 5 words) starting with trigger
        if len(clean_q.split()) < 5 and any(clean_q.startswith(t) for t in casual_triggers):
            return True

        return False

    def generate_lightweight_response(self, query: str) -> str:
        """
        Fast, direct API call for greetings.
        Uses 'gemma-3-27b-it' as requested for lightweight tasks.
        """
        prompt = f"""
        You are Medi-Agent, a helpful and professional clinical assistant.
        The user said: "{query}"

        INSTRUCTIONS:
        1. Respond warmly and politely.
        2. Keep it short (1-2 sentences).
        3. Do NOT provide medical advice yet.
        4. Simply acknowledge and offer help.
        """

        try:
            # FAST TRACK MODEL: GEMMA 3 (Instruction Tuned)
            response = self.client.models.generate_content(
                model='gemma-3-27b-it',
                contents=prompt
            )
            return response.text.strip()
        except Exception:
            return "Hello! I am ready to assist you with your clinical questions."

    def orchestrate_response(self, context: str, query: str, chat_history: list, is_patient_mode: bool = False,
                             status_callback=None) -> str:
        """
        Orchestrates the multi-agent workflow:
        1. Researcher Agent: Extracts raw facts (Model: gemini-3-flash-preview).
        2. Writer Agent: Formats based on 'is_patient_mode' (Model: gemini-3-flash-preview).
        """

        # --- AGENT 1: THE RESEARCHER (Fact Finder) ---
        if status_callback:
            status_callback("🕵️ Researcher Agent: Extracting strict clinical facts...")

        researcher_prompt = f"""
        ROLE: Clinical Researcher
        TASK: Extract relevant medical facts, numbers, and protocols from the provided text to answer the User Query.
        CONSTRAINTS: 
        - Be purely objective. No conversational filler.
        - Extract exact dosages, contraindications, and rule differences.
        - Retain [Source: "filename", Page: X] metadata for every fact.
        - If region-specific rules exist (e.g., NE India vs Rest), separate them clearly.

        USER QUERY: {query}
        MEDICAL CONTEXT: {context}
        """

        try:
            # HEAVY LIFTING MODEL: GEMINI 3 FLASH PREVIEW
            research_response = self.client.models.generate_content(
                model='gemini-3-flash-preview',
                contents=researcher_prompt
            )
            research_notes = research_response.text
        except Exception as e:
            return f"Researcher Error: {str(e)}"

        # --- AGENT 2: THE WRITER (Communicator) ---
        if status_callback:
            if is_patient_mode:
                status_callback("❤️ Writer Agent: Translating to patient-friendly language...")
            else:
                status_callback("✍️ Writer Agent: Synthesizing concise clinical summary...")

        # Select Prompt based on Mode
        if is_patient_mode:
            # --- PATIENT MODE (CONCISE) ---
            writer_prompt = f"""
            ROLE: Empathetic Medical Assistant
            TASK: Explain the medical facts to a patient (5th-grade reading level).
            INPUT FACTS: {research_notes}
            USER QUERY: {query}

            GUIDELINES:
            1. **BE BRIEF:** Limit answer to 3-4 short paragraphs maximum.
            2. **STRUCTURE:** Use simple headers like "What is it?", "Symptoms", "Treatment".
            3. **NO FLUFF:** Avoid scary statistics unless necessary.
            4. **REASSURE:** Focus on actionable advice.
            5. Use simple analogies.
            6. Keep citations subtle [Source: "file", Page: X].
            """
        else:
            # --- CLINICIAN MODE (EXECUTIVE SUMMARY) ---
            writer_prompt = f"""
            ROLE: Clinical Decision Support
            TASK: Provide a Direct, Executive-Style Answer.
            INPUT FACTS: {research_notes}
            USER QUERY: {query}

            STRICT CONSTRAINTS:
            1. **BOTTOM LINE UP FRONT:** Start immediately with the answer.
            2. **EXTREME CONCISENESS:** Use sentence fragments or tight bullet points.
            3. **RELEVANCE ONLY:** Do NOT summarize the document. Only answer the specific question.
            4. **CITATION:** You MUST include [Source: "filename", Page: X] for verification.
            """

        formatted_history = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history])

        full_writer_prompt = f"""
        {writer_prompt}

        CHAT HISTORY:
        {formatted_history}
        """

        try:
            # HEAVY LIFTING MODEL: GEMINI 3 FLASH PREVIEW
            final_response = self.client.models.generate_content(
                model='gemini-3-flash-preview',
                contents=full_writer_prompt
            )
            return final_response.text.strip()
        except Exception as e:
            return f"Writer Error: {str(e)}"
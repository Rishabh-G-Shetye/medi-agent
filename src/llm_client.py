from google import genai
from src.config import Config


class GeminiClient:
    def __init__(self):
        # Initialize the new SDK Client
        self.client = genai.Client(api_key=Config.GEMINI_API_KEY)

    def orchestrate_response(self, context: str, query: str, chat_history: list, status_callback=None) -> str:
        """
        Orchestrates the multi-agent workflow:
        1. Researcher Agent analyzes data and extracts facts.
        2. Writer Agent drafts the empathetic, cited response.
        """

        # --- AGENT 1: THE RESEARCHER ---
        if status_callback:
            status_callback("🕵️ Researcher Agent: Analyzing clinical guidelines for key facts...")

        researcher_prompt = f"""
        ROLE: Clinical Researcher
        TASK: Analyze the provided Medical Context and extract key clinical facts relevant to the User Query.
        CONSTRAINTS: 
        - Be purely objective and factual.
        - Do not be conversational. 
        - List findings in bullet points.
        - If the context mentions specific dosages or contraindications, capture them exactly.
        - Retain the Source File and Page Number metadata for every fact.

        USER QUERY: {query}

        MEDICAL CONTEXT:
        {context}
        """

        try:
            # Call LLM for Research Step using the NEW SDK syntax
            research_response = self.client.models.generate_content(
                model='gemini-1.5-flash',
                contents=researcher_prompt
            )
            research_notes = research_response.text
        except Exception as e:
            return f"Researcher Agent Error: {str(e)}"

        # --- AGENT 2: THE WRITER ---
        if status_callback:
            status_callback("✍️ Writer Agent: Synthesizing notes and drafting clinical response...")

        # Format history for context
        formatted_history = "\n".join(
            [f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history]
        )

        writer_prompt = f"""
        ROLE: Medical Communicator (Medi-Agent)
        TASK: Synthesize the RESEARCHER NOTES below into a clear, professional response for a doctor.

        INPUT DATA (From Researcher):
        {research_notes}

        USER QUERY: {query}

        CHAT HISTORY:
        {formatted_history}

        GUIDELINES:
        1. Use a professional but accessible medical tone.
        2. STRICTLY use the facts found by the Researcher.
        3. CITATION RULE: You MUST cite the [Source: "filename", Page: X] for every medical fact.
        4. If the Researcher notes say "No information found", state that clearly.
        5. Format with clear headings and bullet points where appropriate.
        """

        try:
            # Call LLM for Writing Step using the NEW SDK syntax
            final_response = self.client.models.generate_content(
                model='gemini-1.5-flash',
                contents=writer_prompt
            )
            return final_response.text.strip()
        except Exception as e:
            return f"Writer Agent Error: {str(e)}"
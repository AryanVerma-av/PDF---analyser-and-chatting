from app.config import settings


class GeneratorError(Exception):
    """Exception raised when LLM generation fails."""
    pass


class LLMGenerator:
    """Generates grounded answers from retrieved context using Google Gemini or OpenAI."""

    SYSTEM_PROMPT = (
        "You are a professional PDF question-answering assistant.\n"
        "Your task is to answer the user's question based strictly on the provided context retrieved from an uploaded PDF.\n\n"
        "Strict rules:\n"
        "1. Ground your response entirely on the retrieved context.\n"
        "2. If the context does not contain sufficient information to answer the question, clearly state: "
        "'The provided document does not contain sufficient information to answer this question.'\n"
        "3. Do not extrapolate, speculate, or fabricate any facts.\n"
        "4. When citing information, mention the page numbers indicated in the context (e.g., 'According to Page 2...').\n"
        "5. Keep the response concise, factual, and well-structured.\n"
        "6. Do NOT use any emojis in your answer under any circumstances."
    )

    def _generate_gemini(self, question: str, context: str) -> str:
        if not settings.GEMINI_API_KEY:
            raise GeneratorError(
                "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in backend/.env"
            )

        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)

            model_name = settings.GEMINI_LLM_MODEL
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=self.SYSTEM_PROMPT,
                    generation_config={"temperature": 0.1},
                )
                prompt = f"Retrieved Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
                response = model.generate_content(prompt)
            except Exception as model_err:
                if "404" in str(model_err) or "not found" in str(model_err).lower():
                    fallback_model = "gemini-flash-latest" if model_name != "gemini-flash-latest" else "gemini-3.6-flash"
                    model = genai.GenerativeModel(
                        model_name=fallback_model,
                        system_instruction=self.SYSTEM_PROMPT,
                        generation_config={"temperature": 0.1},
                    )
                    prompt = f"Retrieved Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
                    response = model.generate_content(prompt)
                else:
                    raise model_err

            return (response.text or "").strip()
        except Exception as exc:
            raise GeneratorError(f"Google Gemini generation failed: {str(exc)}") from exc

    def _generate_openai(self, question: str, context: str) -> str:
        if not settings.OPENAI_API_KEY:
            raise GeneratorError(
                "OPENAI_API_KEY is not configured. Please set OPENAI_API_KEY or GEMINI_API_KEY in backend/.env"
            )

        try:
            from openai import OpenAI, OpenAIError
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            user_content = f"Retrieved Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

            response = client.chat.completions.create(
                model=settings.OPENAI_LLM_MODEL,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            answer = response.choices[0].message.content or ""
            return answer.strip()
        except Exception as exc:
            raise GeneratorError(f"OpenAI completion failed: {str(exc)}") from exc

    def _generate_groq(self, question: str, context: str) -> str:
        if not settings.GROQ_API_KEY:
            raise GeneratorError(
                "GROQ_API_KEY is not configured. Please set GROQ_API_KEY in backend/.env"
            )

        try:
            from groq import Groq
            client = Groq(api_key=settings.GROQ_API_KEY)
            user_content = f"Retrieved Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

            models_to_try = [settings.GROQ_LLM_MODEL]
            for fallback in ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]:
                if fallback not in models_to_try:
                    models_to_try.append(fallback)

            last_exc = None
            for model_name in models_to_try:
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        temperature=0.1,
                        messages=[
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                        ],
                    )
                    answer = response.choices[0].message.content or ""
                    return answer.strip()
                except Exception as m_exc:
                    last_exc = m_exc
                    continue

            if last_exc:
                raise last_exc
            raise GeneratorError("No models were available to fulfill the Groq request.")
        except Exception as exc:
            raise GeneratorError(f"Groq completion failed: {str(exc)}") from exc

    def generate(self, question: str, context: str) -> str:
        """Sends the question and retrieved context to the active LLM provider."""
        if not context.strip():
            return "No relevant information could be retrieved from the document to answer this question."

        provider = settings.ACTIVE_PROVIDER
        if provider == "groq":
            return self._generate_groq(question, context)
        elif provider == "gemini":
            return self._generate_gemini(question, context)
        return self._generate_openai(question, context)


import ollama

from offline_ai_assistant.config import settings


class OllamaLLM:
    """Handles local LLM communication using Ollama."""

    def __init__(self):
        self.client = ollama.Client(
            host=settings.ollama_base_url,
            timeout=settings.ollama_timeout,
        )

        self.model = settings.ollama_model

    def check_connection(self) -> bool:
        """Check whether Ollama is accessible."""

        try:
            self.client.list()
            return True

        except Exception as error:
            print(f"Ollama connection failed: {error}")
            return False

    def generate_response(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful offline AI assistant.",
        temperature: float | None = None,
    ) -> str:
        """Generate a response using the local LLM."""

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                options={
                    "temperature": (
                        temperature
                        if temperature is not None
                        else settings.llm_temperature
                    )
                },
            )

            return response["message"]["content"]

        except Exception as error:
            raise RuntimeError(
                f"Failed to generate LLM response: {error}"
            ) from error

    def generate_chat_response(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """Generate a response using conversation context."""

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": settings.llm_temperature,
                },
            )

            return response["message"]["content"]

        except Exception as error:
            raise RuntimeError(
                f"Failed to generate chat response: {error}"
            ) from error


# Reusable LLM instance
llm = OllamaLLM()


# Manage chat interactions

from offline_ai_assistant.core.llm import llm
from offline_ai_assistant.assistant.memory import ConversationMemory


class ChatEngine:
    """Manages conversations with selectable local chat models."""

    def __init__(
        self,
        system_prompt: str | None = None,
        model_manager=None,
    ):
        self.memory = ConversationMemory(max_messages=20)
        self.model_manager = model_manager

        self.system_prompt = (
            system_prompt
            or "You are a helpful offline AI assistant. "
            "Provide clear, accurate and concise answers."
        )

    def send_message_with_metrics(
        self,
        user_message: str,
        model_name: str | None = None,
    ) -> dict:
        """Generate a response and return its model metrics."""

        if not user_message or not user_message.strip():
            raise ValueError("Message cannot be empty.")

        user_message = user_message.strip()

        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            *self.memory.get_history(),
            {
                "role": "user",
                "content": user_message,
            },
        ]

        try:
            if self.model_manager is not None:
                result = self.model_manager.generate(
                    messages=messages,
                    model_name=model_name,
                )

                response = result["response"]

                model = result["model"]
                metrics = result["metrics"]

            else:
                # Backward-compatible fallback.
                response = llm.generate_chat_response(messages)

                model = llm.model
                metrics = {}

            # Save history only after successful generation.
            self.memory.add_message("user", user_message)
            self.memory.add_message("assistant", response)

            return {
                "response": response,
                "model": model,
                "metrics": metrics,
            }

        except Exception as error:
            raise RuntimeError(
                f"Chat generation failed: {error}"
            ) from error


    def stream_message(
        self,
        user_message: str,
        model_name: str | None = None,
    ):
        """Stream a response while preserving conversation memory."""

        if not user_message or not user_message.strip():
            raise ValueError("Message cannot be empty.")

        user_message = user_message.strip()

        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            *self.memory.get_history(),
            {
                "role": "user",
                "content": user_message,
            },
        ]

        if self.model_manager is None:
            raise RuntimeError(
                "Streaming requires the configured ModelManager."
            )

        response_parts = []

        try:
            for chunk in self.model_manager.generate_stream(
                messages=messages,
                model_name=model_name,
            ):
                response_parts.append(chunk)
                yield chunk

            response = "".join(response_parts)

            if not response:
                raise RuntimeError("The model returned an empty response.")

            self.memory.add_message("user", user_message)
            self.memory.add_message("assistant", response)

        except Exception as error:
            raise RuntimeError(
                f"Chat streaming failed: {error}"
            ) from error


    def send_message(self, user_message: str) -> str:
        """Backward-compatible chat method returning response text."""

        result = self.send_message_with_metrics(user_message)

        return result["response"]

    def reset(self) -> None:
        """Reset conversation history."""

        self.memory.clear()

    def get_history(self) -> list[dict[str, str]]:
        """Return conversation history."""

        return self.memory.get_history()

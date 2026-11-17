# Manage chat interactions

from offline_ai_assistant.core.llm import llm
from offline_ai_assistant.assistant.memory import ConversationMemory


class ChatEngine:
    """Manages conversations with the local LLM."""

    def __init__(self, system_prompt: str | None = None):
        self.memory = ConversationMemory(max_messages=20)

        self.system_prompt = (
            system_prompt
            or "You are a helpful offline AI assistant. "
               "Provide clear, accurate and concise answers."
        )

    def send_message(self, user_message: str) -> str:
        """Send a message and generate a context-aware response."""

        if not user_message.strip():
            raise ValueError("Message cannot be empty.")

        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            *self.memory.get_history(),
            {
                "role": "user",
                "content": user_message.strip(),
            },
        ]

        try:
            response = llm.generate_chat_response(messages)

            # Save messages only after successful generation
            self.memory.add_message("user", user_message.strip())
            self.memory.add_message("assistant", response)

            return response

        except Exception as error:
            raise RuntimeError(
                f"Chat generation failed: {error}"
            ) from error

    def reset(self) -> None:
        """Reset conversation history."""

        self.memory.clear()

    def get_history(self) -> list[dict[str, str]]:
        """Return conversation history."""

        return self.memory.get_history()

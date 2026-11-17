# It stores conversation history and manages context.

from typing import Literal


class ConversationMemory:
    """Stores and manages conversation history."""

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.history: list[dict[str, str]] = []

    def add_message(
        self,
        role: Literal["user", "assistant"],
        content: str,
    ) -> None:
        """Add a message to conversation history."""

        self.history.append({
            "role": role,
            "content": content,
        })

        # Keep only the latest messages
        if len(self.history) > self.max_messages:
            self.history = self.history[-self.max_messages:]

    def get_history(self) -> list[dict[str, str]]:
        """Return conversation history."""

        return self.history.copy()

    def clear(self) -> None:
        """Clear conversation history."""

        self.history.clear()

    def get_message_count(self) -> int:
        """Return total stored messages."""

        return len(self.history)

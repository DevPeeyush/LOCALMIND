
"""
Hybrid session manager for the Offline AI Assistant.

Responsibilities:
- Manage multiple independent conversations.
- Route policy questions through RAG.
- Route general questions through Ollama.
- Preserve conversational context.
- Restore conversation history.
- Persist all exchanges using SQLite.
"""

import re

from offline_ai_assistant.assistant.chat import ChatEngine
from offline_ai_assistant.assistant.conversation_store import (
    ConversationStore,
)


class SessionManager:
    """Coordinates sessions, RAG and general AI chat."""

    # Terms associated with the Share India policy knowledge base.
    POLICY_PATTERNS = [
        r"\bshare\s+india\b",
        r"\bcompany\s+polic(?:y|ies)\b",
        r"\bemployee\s+benefits?\b",
        r"\bbenefits?\b",
        r"\bhr\s+polic(?:y|ies)\b",
        r"\bemployee\b",
        r"\bworkplace\b",
        r"\bmaternity\b",
        r"\bposh\b",
        r"\bharassment\b",
        r"\bgrievance\b",
        r"\bhuman\s+rights\b",
        r"\bequal\s+opportunity\b",
        r"\bworkplace\s+safety\b",
        r"\bmedical\s+insurance\b",
        r"\binsurance\s+coverage\b",
        r"\b(?:sick|casual|earned|annual|paid|unpaid)\s+leave\b",
        r"\bleave\s+policy\b",
        r"\bleave\s+entitlement\b",
        r"\bcafeteria\b",
        r"\btransport\s+facility\b",
        r"\bemployee\s+wellness\b",
        r"\bcompany\s+transport\b",
    ]

    FOLLOW_UP_PATTERNS = [
        r"^\s*what\s+about\b",
        r"^\s*how\s+about\b",
        r"^\s*and\s+what\b",
        r"^\s*what\s+if\b",
        r"^\s*how\s+many\b",
        r"^\s*how\s+much\b",
        r"^\s*is\s+it\b",
        r"^\s*does\s+it\b",
        r"^\s*can\s+i\b",
        r"^\s*am\s+i\b",
        r"^\s*what\s+are\s+the\b",
        r"^\s*and\s+eligibility\b",
        r"^\s*eligibility\b",
    ]

    EXCLUDED_GENERAL_MESSAGES = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "good morning",
        "good afternoon",
        "good evening",
    }

    def __init__(
        self,
        store: ConversationStore | None = None,
        rag_pipeline=None,
        max_messages: int = 20,
        system_prompt: str | None = None,
    ):
        self.store = store or ConversationStore()

        self.rag_pipeline = rag_pipeline

        self.max_messages = max_messages
        self.system_prompt = system_prompt

        # Cache independent ChatEngine instances by conversation ID.
        self._engines: dict[str, ChatEngine] = {}

    # --------------------------------------------------
    # ENGINE MANAGEMENT
    # --------------------------------------------------

    def _create_engine(self) -> ChatEngine:
        """Create a fresh general-purpose chat engine."""

        engine = ChatEngine(
            system_prompt=self.system_prompt
        )

        engine.memory.max_messages = self.max_messages

        return engine

    def _get_engine(self, conversation_id: str) -> ChatEngine:
        """Get an existing engine or restore its history."""

        engine = self._engines.get(conversation_id)

        if engine is not None:
            return engine

        self.load_conversation(conversation_id)

        return self._engines[conversation_id]

    # --------------------------------------------------
    # HYBRID ROUTING
    # --------------------------------------------------

    @classmethod
    def _is_policy_question(cls, message: str) -> bool:
        """Identify questions related to company policies."""

        normalized = message.lower().strip()

        if normalized in cls.EXCLUDED_GENERAL_MESSAGES:
            return False

        return any(
            re.search(pattern, normalized)
            for pattern in cls.POLICY_PATTERNS
        )

    @classmethod
    def _is_follow_up(cls, message: str) -> bool:
        """Identify short contextual follow-up questions."""

        normalized = message.lower().strip()

        return any(
            re.search(pattern, normalized)
            for pattern in cls.FOLLOW_UP_PATTERNS
        )

    def _get_last_user_message(
        self,
        messages: list[dict],
    ) -> str | None:
        """Find the most recent user message."""

        for message in reversed(messages):
            if message["role"] == "user":
                return message["content"]

        return None

    def _resolve_message_mode(
        self,
        message: str,
        history: list[dict],
    ) -> tuple[str, str]:
        """
        Resolve whether to use RAG or general Ollama chat.

        Returns:
            mode, effective_query
        """

        if self._is_policy_question(message):
            return "rag", message

        last_user_message = self._get_last_user_message(
            history
        )

        # Preserve policy context for follow-up questions.
        if (
            last_user_message
            and self._is_follow_up(message)
            and self._is_policy_question(last_user_message)
        ):
            effective_query = (
                f"Previous policy question: {last_user_message}\n"
                f"Follow-up question: {message}"
            )

            return "rag", effective_query

        return "general", message

    # --------------------------------------------------
    # CONVERSATION MANAGEMENT
    # --------------------------------------------------

    def create_conversation(
        self,
        title: str = "New conversation",
    ) -> dict:
        """Create a new independent conversation."""

        conversation = self.store.create_conversation(title)

        self._engines[conversation["id"]] = (
            self._create_engine()
        )

        return conversation

    def load_conversation(
        self,
        conversation_id: str,
    ) -> dict:
        """Restore a conversation and its messages."""

        conversation = self.store.get_conversation(
            conversation_id
        )

        if conversation is None:
            raise ValueError("Conversation does not exist.")

        messages = self.store.get_messages(
            conversation_id
        )

        engine = self._create_engine()

        for message in messages:
            engine.memory.add_message(
                role=message["role"],
                content=message["content"],
            )

        self._engines[conversation_id] = engine

        return {
            "conversation": conversation,
            "messages": messages,
        }

    def list_conversations(self) -> list[dict]:
        """List all conversations."""

        return self.store.list_conversations()

    def get_conversation(self, conversation_id: str) -> dict:
        """Get conversation metadata and complete history."""

        conversation = self.store.get_conversation(
            conversation_id
        )

        if conversation is None:
            raise ValueError("Conversation does not exist.")

        return {
            "conversation": conversation,
            "messages": self.store.get_messages(
                conversation_id
            ),
        }

    # --------------------------------------------------
    # CHAT OPERATIONS
    # --------------------------------------------------

    def send_message(
        self,
        conversation_id: str,
        user_message: str,
    ) -> dict:
        """Route, generate and persist a chat response."""

        if not user_message or not user_message.strip():
            raise ValueError("Message cannot be empty.")

        user_message = user_message.strip()

        conversation = self.store.get_conversation(
            conversation_id
        )

        if conversation is None:
            raise ValueError("Conversation does not exist.")

        if (
            self.rag_pipeline is None
            and self._is_policy_question(user_message)
        ):
            raise RuntimeError(
                "RAG pipeline is not configured. "
                "Cannot process policy-related questions."
            )

        engine = self._get_engine(conversation_id)

        previous_messages = self.store.get_messages(
            conversation_id
        )

        mode, effective_query = self._resolve_message_mode(
            user_message,
            engine.get_history(),
        )

        # ----------------------------------------------
        # Generate response using the selected engine.
        # ----------------------------------------------

        if mode == "rag":

            if self.rag_pipeline is None:
                raise RuntimeError(
                    "RAG pipeline is unavailable."
                )

            rag_result = self.rag_pipeline.run(
                effective_query
            )

            response = rag_result["answer"]

            sources = rag_result.get("sources", [])

            # Update in-memory conversation history.
            engine.memory.add_message(
                "user",
                user_message,
            )

            engine.memory.add_message(
                "assistant",
                response,
            )

        else:

            response = engine.send_message(
                user_message
            )

            sources = []

        # ----------------------------------------------
        # Persist the completed exchange.
        # ----------------------------------------------

        try:
            self.store.add_messages(
                conversation_id,
                [
                    {
                        "role": "user",
                        "content": user_message,
                    },
                    {
                        "role": "assistant",
                        "content": response,
                    },
                ],
            )

            # Generate a title from the first user message.
            if not previous_messages:

                title = user_message.replace("\n", " ")

                if len(title) > 50:
                    title = title[:47].rstrip() + "..."

                self.store.rename_conversation(
                    conversation_id,
                    title,
                )

        except Exception as error:

            self._engines.pop(conversation_id, None)

            raise RuntimeError(
                f"Failed to persist conversation: {error}"
            ) from error

        return {
            "conversation_id": conversation_id,
            "mode": mode,
            "question": user_message,
            "response": response,
            "sources": sources,
            "messages": engine.get_history(),
        }

    # --------------------------------------------------
    # HISTORY AND RESET
    # --------------------------------------------------

    def get_history(
        self,
        conversation_id: str,
    ) -> list[dict]:
        """Return persistent conversation history."""

        if self.store.get_conversation(conversation_id) is None:
            raise ValueError("Conversation does not exist.")

        return self.store.get_messages(
            conversation_id
        )

    def reset_conversation(
        self,
        conversation_id: str,
    ) -> bool:
        """Clear conversation messages."""

        if self.store.get_conversation(conversation_id) is None:
            raise ValueError("Conversation does not exist.")

        self.store.clear_messages(conversation_id)

        engine = self._engines.get(conversation_id)

        if engine is not None:
            engine.reset()

        return True

    # --------------------------------------------------
    # RENAME AND DELETE
    # --------------------------------------------------

    def rename_conversation(
        self,
        conversation_id: str,
        new_title: str,
    ) -> bool:
        """Rename a conversation."""

        return self.store.rename_conversation(
            conversation_id,
            new_title,
        )

    def delete_conversation(
        self,
        conversation_id: str,
    ) -> bool:
        """Delete a conversation and its cached engine."""

        deleted = self.store.delete_conversation(
            conversation_id
        )

        if deleted:
            self._engines.pop(conversation_id, None)

        return deleted

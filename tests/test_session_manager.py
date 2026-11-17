
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from offline_ai_assistant.assistant.conversation_store import (
    ConversationStore,
)
from offline_ai_assistant.assistant.session_manager import (
    SessionManager,
)


def test_session_manager():

    with TemporaryDirectory() as temp_dir:

        db_path = Path(temp_dir) / "test_conversations.db"

        store = ConversationStore(db_path=db_path)

        manager = SessionManager(store=store)

        # 1. Create first conversation
        conversation1 = manager.create_conversation()

        id1 = conversation1["id"]

        print("\nConversation 1:", id1)

        # 2. Generate a response using mocked Ollama
        with patch(
            "offline_ai_assistant.assistant.chat.llm.generate_chat_response",
            return_value="Share India provides employee benefits."
        ):
            result = manager.send_message(
                id1,
                "What employee benefits does Share India provide?"
            )

        print("\nAI Response:", result["response"])

        # 3. Verify automatic title generation
        conversation = store.get_conversation(id1)

        assert conversation["title"] == (
            "What employee benefits does Share India provide?"
        )

        # 4. Create another conversation
        conversation2 = manager.create_conversation(
            "Second conversation"
        )

        id2 = conversation2["id"]

        assert id1 != id2

        # 5. Verify conversation isolation
        assert len(manager.get_history(id1)) == 2
        assert len(manager.get_history(id2)) == 0

        # 6. Simulate application restart
        new_manager = SessionManager(
            store=ConversationStore(db_path=db_path)
        )

        restored = new_manager.load_conversation(id1)

        assert len(restored["messages"]) == 2

        assert restored["messages"][0]["role"] == "user"
        assert restored["messages"][1]["role"] == "assistant"

        # 7. Verify history restoration
        history = new_manager.get_history(id1)

        assert history[0]["content"] == (
            "What employee benefits does Share India provide?"
        )

        # 8. Verify conversation listing
        conversations = new_manager.list_conversations()

        assert len(conversations) == 2

        # 9. Verify reset
        new_manager.reset_conversation(id1)

        assert len(new_manager.get_history(id1)) == 0

        # 10. Verify deletion
        assert new_manager.delete_conversation(id2)

        assert len(new_manager.list_conversations()) == 1

        print("\nAll SessionManager tests passed!")


if __name__ == "__main__":
    test_session_manager()

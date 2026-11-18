
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

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

        # Mock RAG pipeline because this test uses a policy question.
        mock_rag_pipeline = Mock()

        mock_rag_pipeline.run.return_value = {
            "question": "What employee benefits does Share India provide?",
            "answer": "Share India provides employee benefits. [S1]",
            "sources": [
                {
                    "source_id": "S1",
                    "file_name": "employee_benefits.pdf",
                    "page": 1,
                }
            ],
        }

        manager = SessionManager(
            store=store,
            rag_pipeline=mock_rag_pipeline,
        )

        # 1. Create first conversation
        conversation1 = manager.create_conversation()

        id1 = conversation1["id"]

        print("\nConversation 1:", id1)

        # 2. Generate a response through the mocked RAG pipeline
        result = manager.send_message(
            id1,
            "What employee benefits does Share India provide?",
        )

        print("\nAI Response:", result["response"])

        assert result["mode"] == "rag"
        assert result["response"] == (
            "Share India provides employee benefits. [S1]"
        )

        mock_rag_pipeline.run.assert_called_once()

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

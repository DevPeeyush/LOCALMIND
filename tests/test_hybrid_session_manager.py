
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from offline_ai_assistant.assistant.conversation_store import (
    ConversationStore,
)
from offline_ai_assistant.assistant.session_manager import (
    SessionManager,
)


class FakeRAGPipeline:
    """Mock RAG pipeline for routing tests."""

    def __init__(self):
        self.queries = []

    def run(self, query):
        self.queries.append(query)

        return {
            "question": query,
            "answer": "The retrieved policy information supports this answer. [S1]",
            "sources": [
                {
                    "source_id": "S1",
                    "file_name": "maternity_leave_policy.pdf",
                    "page": 2,
                }
            ],
        }


def test_hybrid_session_manager():

    with TemporaryDirectory() as temp_dir:

        db_path = Path(temp_dir) / "hybrid_test.db"

        store = ConversationStore(db_path=db_path)
        rag = FakeRAGPipeline()

        manager = SessionManager(
            store=store,
            rag_pipeline=rag,
        )

        conversation = manager.create_conversation()
        conversation_id = conversation["id"]

        # ------------------------------------------
        # 1. General question should use Ollama.
        # ------------------------------------------

        with patch(
            "offline_ai_assistant.assistant.chat.llm.generate_chat_response",
            return_value="Python decorators modify function behaviour."
        ) as mock_llm:

            general_result = manager.send_message(
                conversation_id,
                "Explain Python decorators."
            )

        assert general_result["mode"] == "general"
        assert general_result["sources"] == []
        mock_llm.assert_called_once()

        # ------------------------------------------
        # 2. Policy question should use RAG.
        # ------------------------------------------

        policy_result = manager.send_message(
            conversation_id,
            "What maternity leave policy applies?"
        )

        assert policy_result["mode"] == "rag"
        assert len(policy_result["sources"]) == 1
        assert len(rag.queries) == 1

        # ------------------------------------------
        # 3. Follow-up should preserve policy context.
        # ------------------------------------------

        followup_result = manager.send_message(
            conversation_id,
            "How many days?"
        )

        assert followup_result["mode"] == "rag"

        assert "Previous policy question" in rag.queries[1]
        assert "How many days?" in rag.queries[1]

        # ------------------------------------------
        # 4. Verify persistence.
        # ------------------------------------------

        history = manager.get_history(conversation_id)

        assert len(history) == 6

        # ------------------------------------------
        # 5. Verify source isolation.
        # ------------------------------------------

        assert general_result["sources"] == []
        assert policy_result["sources"][0]["source_id"] == "S1"

        print("\nGeneral routing: PASSED")
        print("Policy routing: PASSED")
        print("Follow-up routing: PASSED")
        print("Source handling: PASSED")
        print("Conversation persistence: PASSED")

        print("\nAll hybrid routing tests passed!")


if __name__ == "__main__":
    test_hybrid_session_manager()

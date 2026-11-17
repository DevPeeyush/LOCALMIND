
"""
Persistent conversation storage using SQLite.

Responsibilities:
- Create and manage conversations.
- Store user and assistant messages.
- Retrieve conversation history.
- Rename and delete conversations.
- Maintain conversation timestamps.
"""

import sqlite3
import uuid

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


# Project root: Offline AI Assistant/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "conversations.db"


class ConversationStore:
    """Handles persistent conversation storage using SQLite."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._initialize_database()

    @staticmethod
    def _get_timestamp() -> str:
        """Return current UTC timestamp."""
        return datetime.now(timezone.utc).isoformat()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Create and safely close a database connection."""

        connection = sqlite3.connect(
            str(self.db_path),
            timeout=30,
        )

        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")

        try:
            yield connection
            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    def _initialize_database(self) -> None:
        """Create database tables if they do not exist."""

        with self._get_connection() as connection:

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT 'New conversation',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL
                        CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,

                    FOREIGN KEY(conversation_id)
                    REFERENCES conversations(id)
                    ON DELETE CASCADE
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_messages_conversation_id
                ON messages(conversation_id)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_conversations_updated_at
                ON conversations(updated_at)
                """
            )

    # --------------------------------------------------
    # CONVERSATION OPERATIONS
    # --------------------------------------------------

    def create_conversation(
        self,
        title: str = "New conversation",
    ) -> dict:
        """Create a new conversation."""

        conversation_id = str(uuid.uuid4())
        timestamp = self._get_timestamp()

        with self._get_connection() as connection:

            connection.execute(
                """
                INSERT INTO conversations (
                    id, title, created_at, updated_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    title.strip() or "New conversation",
                    timestamp,
                    timestamp,
                ),
            )

        return self.get_conversation(conversation_id)

    def get_conversation(self, conversation_id: str) -> dict | None:
        """Retrieve conversation metadata."""

        with self._get_connection() as connection:

            row = connection.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM conversations
                WHERE id = ?
                """,
                (conversation_id,),
            ).fetchone()

        return dict(row) if row else None

    def list_conversations(self) -> list[dict]:
        """Return all conversations, newest activity first."""

        with self._get_connection() as connection:

            rows = connection.execute(
                """
                SELECT
                    c.id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m
                    ON c.id = m.conversation_id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                """
            ).fetchall()

        return [dict(row) for row in rows]

    def rename_conversation(
        self,
        conversation_id: str,
        new_title: str,
    ) -> bool:
        """Rename an existing conversation."""

        if not new_title.strip():
            raise ValueError("Conversation title cannot be empty.")

        with self._get_connection() as connection:

            cursor = connection.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    new_title.strip(),
                    self._get_timestamp(),
                    conversation_id,
                ),
            )

        return cursor.rowcount > 0

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its messages."""

        with self._get_connection() as connection:

            cursor = connection.execute(
                """
                DELETE FROM conversations
                WHERE id = ?
                """,
                (conversation_id,),
            )

        return cursor.rowcount > 0

    # --------------------------------------------------
    # MESSAGE OPERATIONS
    # --------------------------------------------------

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> dict:
        """Save a message to a conversation."""

        if role not in ("user", "assistant"):
            raise ValueError("Role must be user or assistant.")

        if not content.strip():
            raise ValueError("Message content cannot be empty.")

        timestamp = self._get_timestamp()

        with self._get_connection() as connection:

            conversation = connection.execute(
                """
                SELECT id FROM conversations
                WHERE id = ?
                """,
                (conversation_id,),
            ).fetchone()

            if conversation is None:
                raise ValueError("Conversation does not exist.")

            cursor = connection.execute(
                """
                INSERT INTO messages (
                    conversation_id,
                    role,
                    content,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    role,
                    content,
                    timestamp,
                ),
            )

            message_id = cursor.lastrowid

            connection.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ?
                """,
                (timestamp, conversation_id),
            )

        return {
            "id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "created_at": timestamp,
        }

    def get_messages(
        self,
        conversation_id: str,
    ) -> list[dict]:
        """Retrieve messages in chronological order."""

        with self._get_connection() as connection:

            rows = connection.execute(
                """
                SELECT
                    id,
                    conversation_id,
                    role,
                    content,
                    created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (conversation_id,),
            ).fetchall()

        return [dict(row) for row in rows]


    def add_messages(
        self,
        conversation_id: str,
        messages: list[dict[str, str]],
    ) -> list[dict]:
        """Save multiple messages in a single transaction."""

        if not messages:
            return []

        for message in messages:
            if message.get("role") not in ("user", "assistant"):
                raise ValueError("Invalid message role.")

            if not message.get("content", "").strip():
                raise ValueError("Message content cannot be empty.")

        timestamp = self._get_timestamp()

        with self._get_connection() as connection:

            exists = connection.execute(
                "SELECT id FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()

            if exists is None:
                raise ValueError("Conversation does not exist.")

            connection.executemany(
                """
                INSERT INTO messages (
                    conversation_id, role, content, created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        conversation_id,
                        message["role"],
                        message["content"],
                        timestamp,
                    )
                    for message in messages
                ],
            )

            connection.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ?
                """,
                (timestamp, conversation_id),
            )

        return self.get_messages(conversation_id)


    def clear_messages(self, conversation_id: str) -> bool:
        """Delete all messages without deleting the conversation."""

        with self._get_connection() as connection:

            cursor = connection.execute(
                """
                DELETE FROM messages
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            )

            connection.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ?
                """,
                (self._get_timestamp(), conversation_id),
            )

        return cursor.rowcount >= 0

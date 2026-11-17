"""Pydantic schemas for chat API."""
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator

class ChatRequest(BaseModel):
    """Incoming chat request."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Message sent by the user.",
    )

    conversation_id: str | None = Field(
        default=None,
        description="Existing conversation ID. Omit to create a new conversation.",
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Message cannot be empty.")

        return value


class ChatMessage(BaseModel):
    """Individual conversation message."""

    role: Literal["user", "assistant"]
    content: str

    id: int | None = None
    conversation_id: str | None = None
    created_at: str | None = None


class ChatResponse(BaseModel):
    """Response returned by the chat endpoint."""

    conversation_id: str

    mode: Literal["general", "rag"]

    question: str

    response: str

    sources: list[dict[str, Any]] = Field(
        default_factory=list
    )

    messages: list[ChatMessage] = Field(
        default_factory=list
    )


class ChatHistoryResponse(BaseModel):
    """Conversation history response."""

    conversation_id: str
    messages: list[ChatMessage]


class ResetConversationResponse(BaseModel):
    """Conversation reset response."""

    conversation_id: str
    message: str

class ConversationCreateRequest(BaseModel):
    """Request to create a conversation."""

    title: str = Field(
        default="New conversation",
        max_length=100,
    )


class ConversationRenameRequest(BaseModel):
    """Request to rename a conversation."""

    title: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Title cannot be empty.")

        return value


class ConversationResponse(BaseModel):
    """Conversation metadata."""

    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int | None = None


class ConversationActionResponse(BaseModel):
    """Response for conversation management actions."""

    conversation_id: str
    message: str

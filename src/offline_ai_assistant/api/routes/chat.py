
"""Chat API routes."""
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, HTTPException, Request
from offline_ai_assistant.api.schemas.chat import (
    ConversationCreateRequest,
    ConversationRenameRequest,
    ConversationResponse,
    ConversationActionResponse,
)
from offline_ai_assistant.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatHistoryResponse,
    ResetConversationResponse,
)

router = APIRouter(
    prefix="/api/v1",
    tags=["Chat"],
)


def get_session_manager(request: Request):
    """Get the initialized SessionManager."""

    session_manager = getattr(
        request.app.state,
        "session_manager",
        None,
    )

    if session_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Chat service is not initialized.",
        )

    return session_manager


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message",
)
def send_chat_message(
    payload: ChatRequest,
    request: Request,
):
    """
    Send a message to the hybrid AI assistant.

    - General questions use the local Ollama LLM.
    - Company policy questions use the RAG pipeline.
    - Follow-up policy questions preserve context.
    """

    session_manager = get_session_manager(request)

    try:
        # Create a new conversation when no ID is supplied.
        if payload.conversation_id is None:
            conversation = session_manager.create_conversation()
            conversation_id = conversation["id"]
        else:
            conversation_id = payload.conversation_id

        # Generate response using hybrid routing.
        result = session_manager.send_message(
            conversation_id=conversation_id,
            user_message=payload.message,
            model_name=payload.model_name,
        )

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error
        

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

@router.post("/chat/stream", summary="Stream a chat response")
def stream_chat_message(payload: ChatRequest, request: Request):
    """
    Stream the assistant response as newline-delimited JSON events.

    Event types:
    - start: conversation has started
    - token: contains a generated text chunk
    - done: contains the completed response and metadata
    - error: contains an error message
    """

    session_manager = get_session_manager(request)

    # Create a conversation if one was not provided.
    if payload.conversation_id is None:
        conversation = session_manager.create_conversation()
        conversation_id = conversation["id"]
    else:
        conversation_id = payload.conversation_id

    def event_stream():
        try:
            # Notify the frontend that streaming has started.
            yield json.dumps(
                {
                    "type": "start",
                    "conversation_id": conversation_id,
                },
                ensure_ascii=False,
            ) + "\n"

            # Stream response chunks from SessionManager.
            for event in session_manager.stream_message(
                conversation_id=conversation_id,
                user_message=payload.message,
                model_name=payload.model_name,
            ):
                yield json.dumps(
                    event,
                    ensure_ascii=False,
                ) + "\n"

        except Exception as error:
            yield json.dumps(
                {
                    "type": "error",
                    "detail": str(error),
                },
                ensure_ascii=False,
            ) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
    )

@router.get(
    "/chat/{conversation_id}/history",
    response_model=ChatHistoryResponse,
    summary="Get conversation history",
)
def get_chat_history(
    conversation_id: str,
    request: Request,
):
    """Retrieve all messages in a conversation."""

    session_manager = get_session_manager(request)

    try:
        messages = session_manager.get_history(
            conversation_id
        )

        return {
            "conversation_id": conversation_id,
            "messages": messages,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


@router.post(
    "/chat/{conversation_id}/reset",
    response_model=ResetConversationResponse,
    summary="Reset conversation",
)
def reset_chat(
    conversation_id: str,
    request: Request,
):
    """Clear messages while preserving the conversation."""

    session_manager = get_session_manager(request)

    try:
        session_manager.reset_conversation(
            conversation_id
        )

        return {
            "conversation_id": conversation_id,
            "message": "Conversation history cleared successfully.",
        }

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


@router.get(
    "/conversations",
    response_model=list[ConversationResponse],
    summary="List all conversations",
)
def list_conversations(request: Request):
    """Return all saved conversations."""

    session_manager = get_session_manager(request)

    return session_manager.list_conversations()


@router.post(
    "/conversations",
    response_model=ConversationResponse,
    summary="Create a conversation",
)
def create_conversation(
    payload: ConversationCreateRequest,
    request: Request,
):
    """Create a new conversation."""

    session_manager = get_session_manager(request)

    try:
        return session_manager.create_conversation(
            title=payload.title
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@router.patch(
    "/conversations/{conversation_id}",
    response_model=ConversationActionResponse,
    summary="Rename a conversation",
)
def rename_conversation(
    conversation_id: str,
    payload: ConversationRenameRequest,
    request: Request,
):
    """Rename an existing conversation."""

    session_manager = get_session_manager(request)

    try:
        renamed = session_manager.rename_conversation(
            conversation_id=conversation_id,
            new_title=payload.title,
        )

        if not renamed:
            raise HTTPException(
                status_code=404,
                detail="Conversation does not exist.",
            )

        return {
            "conversation_id": conversation_id,
            "message": "Conversation renamed successfully.",
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@router.delete(
    "/conversations/{conversation_id}",
    response_model=ConversationActionResponse,
    summary="Delete a conversation",
)
def delete_conversation(
    conversation_id: str,
    request: Request,
):
    """Delete a conversation and its messages."""

    session_manager = get_session_manager(request)

    deleted = session_manager.delete_conversation(
        conversation_id
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Conversation does not exist.",
        )

    return {
        "conversation_id": conversation_id,
        "message": "Conversation deleted successfully.",
    }


"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from offline_ai_assistant.api.routes.models import router as models_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from offline_ai_assistant.api.routes.documents import (
    router as documents_router,
)
from offline_ai_assistant.api.dependencies import (
    initialize_services,
)

from offline_ai_assistant.api.routes.chat import (
    router as chat_router,
)
from offline_ai_assistant.api.routes.benchmark import (
    router as benchmark_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\nStarting Offline AI Assistant...")

    services = initialize_services()

    for service_name, service in services.items():
        setattr(app.state, service_name, service)

    print("FAISS index loaded.")
    print("RAG pipeline initialized.")
    print("Conversation storage initialized.")
    print("Hybrid session manager initialized.")
    print("Offline AI Assistant is ready.\n")

    yield

    print("\nShutting down Offline AI Assistant...")


app = FastAPI(
    title="Offline AI Assistant",
    description=(
        "Local AI assistant powered by Ollama, "
        "FAISS, RAG and SQLite."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register API routes.
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(models_router)
app.include_router(benchmark_router)

@app.get("/")
def root():
    return {
        "message": "Welcome to Offline AI Assistant",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Offline AI Assistant API",
    }

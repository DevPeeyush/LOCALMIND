
"""Chat model discovery API."""

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(
    prefix="/api/v1",
    tags=["Models"],
)


def get_model_manager(request: Request):
    """Get the initialized ModelManager."""

    manager = getattr(
        request.app.state,
        "model_manager",
        None,
    )

    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="Model manager is not initialized.",
        )

    return manager


@router.get(
    "/models",
    summary="List available chat models",
)
def list_models(request: Request):
    """Return installed chat models and the default model."""

    manager = get_model_manager(request)

    try:
        models = manager.list_chat_models()

        return {
            "models": models,
            "default_model": manager.default_model,
        }

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

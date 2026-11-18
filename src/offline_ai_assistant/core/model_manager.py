
import time
from typing import Any

import ollama

from offline_ai_assistant.config import settings


class ModelManager:
    """Discovers and manages locally installed Ollama chat models."""

    # Embedding-only model families should never appear in chat selection.
    EMBEDDING_FAMILIES = {"bert"}

    EMBEDDING_NAME_MARKERS = (
        "embed",
        "embedding",
    )

    def __init__(self):
        self.client = ollama.Client(
            host=settings.ollama_base_url,
            timeout=settings.ollama_timeout,
        )

        self.default_model = settings.ollama_model

    @staticmethod
    def _get_value(data: Any, key: str, default=None):
        """Read values from either dictionaries or Ollama response objects."""

        if isinstance(data, dict):
            return data.get(key, default)

        return getattr(data, key, default)

    def check_connection(self) -> bool:
        """Check whether the Ollama server is accessible."""

        try:
            self.client.list()
            return True

        except Exception as error:
            print(f"Ollama connection failed: {error}")
            return False

    def list_chat_models(self) -> list[dict]:
        """Return installed models that appear suitable for chat generation."""

        try:
            response = self.client.list()
            models = self._get_value(response, "models", [])

            chat_models = []

            for model in models:
                name = (
                    self._get_value(model, "model")
                    or self._get_value(model, "name")
                )

                if not name:
                    continue

                details = self._get_value(model, "details", {})

                family = self._get_value(details, "family", "") or ""
                normalized_name = name.lower()

                # Exclude embedding-only models.
                if family.lower() in self.EMBEDDING_FAMILIES:
                    continue

                if any(
                    marker in normalized_name
                    for marker in self.EMBEDDING_NAME_MARKERS
                ):
                    continue

                chat_models.append(
                    {
                        "name": name,
                        "size": self._get_value(model, "size", 0),
                        "family": family,
                        "parameter_size": self._get_value(
                            details, "parameter_size", "Unknown"
                        ),
                        "quantization_level": self._get_value(
                            details, "quantization_level", "Unknown"
                        ),
                        "modified_at": str(
                            self._get_value(model, "modified_at", "")
                        ),
                    }
                )

            return chat_models

        except Exception as error:
            raise RuntimeError(
                f"Failed to discover Ollama models: {error}"
            ) from error

    def get_model_names(self) -> list[str]:
        """Return only the names of available chat models."""

        return [
            model["name"]
            for model in self.list_chat_models()
        ]

    def validate_model(self, model_name: str) -> None:
        """Ensure the requested model is an installed chat model."""

        if not model_name or not model_name.strip():
            raise ValueError("Model name cannot be empty.")

        available_models = self.get_model_names()

        if model_name not in available_models:
            raise ValueError(
                f"Model '{model_name}' is not an available chat model. "
                f"Available models: {available_models}"
            )

    def generate(
        self,
        messages: list[dict[str, str]],
        model_name: str | None = None,
        temperature: float | None = None,
    ) -> dict:
        """
        Generate a response using the selected chat model.

        Returns the answer and Ollama's native generation statistics.
        """

        selected_model = model_name or self.default_model

        self.validate_model(selected_model)

        start_time = time.perf_counter()

        try:
            response = self.client.chat(
                model=selected_model,
                messages=messages,
                options={
                    "temperature": (
                        temperature
                        if temperature is not None
                        else settings.llm_temperature
                    )
                },
            )

            elapsed_time = time.perf_counter() - start_time

            message = self._get_value(response, "message", {})
            content = self._get_value(message, "content", "")

            if not content:
                raise RuntimeError("The selected model returned an empty response.")

            eval_count = self._get_value(response, "eval_count", 0) or 0
            eval_duration = self._get_value(
                response, "eval_duration", 0
            ) or 0

            tokens_per_second = (
                eval_count / (eval_duration / 1_000_000_000)
                if eval_duration > 0
                else 0.0
            )

            return {
                "model": selected_model,
                "response": content,
                "metrics": {
                    "total_duration_seconds": (
                        self._get_value(response, "total_duration", 0)
                        / 1_000_000_000
                    ),
                    "wall_time_seconds": elapsed_time,
                    "load_duration_seconds": (
                        self._get_value(response, "load_duration", 0)
                        / 1_000_000_000
                    ),
                    "prompt_eval_count": self._get_value(
                        response, "prompt_eval_count", 0
                    ),
                    "prompt_eval_duration_seconds": (
                        self._get_value(response, "prompt_eval_duration", 0)
                        / 1_000_000_000
                    ),
                    "eval_count": eval_count,
                    "eval_duration_seconds": (
                        eval_duration / 1_000_000_000
                    ),
                    "tokens_per_second": tokens_per_second,
                },
            }

        except Exception as error:
            raise RuntimeError(
                f"Generation failed using '{selected_model}': {error}"
            ) from error


    def generate_stream(
        self,
        messages: list[dict[str, str]],
        model_name: str | None = None,
        temperature: float | None = None,
    ):
        """Stream response chunks from the selected Ollama model."""

        selected_model = model_name or self.default_model
        self.validate_model(selected_model)

        try:
            stream = self.client.chat(
                model=selected_model,
                messages=messages,
                stream=True,
                options={
                    "temperature": (
                        temperature
                        if temperature is not None
                        else settings.llm_temperature
                    )
                },
            )

            for chunk in stream:
                message = self._get_value(chunk, "message", {})
                content = self._get_value(message, "content", "")

                if content:
                    yield content

        except Exception as error:
            raise RuntimeError(
                f"Streaming generation failed using "
                f"'{selected_model}': {error}"
            ) from error

    def generate_response(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful offline AI assistant.",
        model_name: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a response from a single prompt."""

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        result = self.generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model_name=model_name,
            temperature=temperature,
        )

        return result["response"]

    def generate_chat_response(
        self,
        messages: list[dict[str, str]],
        model_name: str | None = None,
    ) -> str:
        """Generate a response using conversation history."""

        result = self.generate(
            messages=messages,
            model_name=model_name,
        )

        return result["response"]


# Reusable model manager
model_manager = ModelManager()

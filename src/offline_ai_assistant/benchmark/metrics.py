
"""Performance metric calculations for model benchmarking."""


def calculate_metrics(generation_metrics: dict) -> dict:
    """Normalize Ollama generation metrics."""

    return {
        "response_time": round(
            generation_metrics.get("wall_time_seconds", 0.0), 4
        ),
        "total_duration": round(
            generation_metrics.get("total_duration_seconds", 0.0), 4
        ),
        "model_load_time": round(
            generation_metrics.get("load_duration_seconds", 0.0), 4
        ),
        "prompt_tokens": generation_metrics.get(
            "prompt_eval_count", 0
        ),
        "generated_tokens": generation_metrics.get(
            "eval_count", 0
        ),
        "tokens_per_second": round(
            generation_metrics.get("tokens_per_second", 0.0), 2
        ),
    }


def calculate_average_metrics(results: list[dict]) -> dict:
    """Calculate average performance across successful test results."""

    if not results:
        return {}

    metric_names = [
        "response_time",
        "total_duration",
        "model_load_time",
        "prompt_tokens",
        "generated_tokens",
        "tokens_per_second",
    ]

    averages = {}

    for metric in metric_names:
        values = [
            result["metrics"][metric]
            for result in results
            if result.get("metrics") and metric in result["metrics"]
        ]

        averages[metric] = (
            round(sum(values) / len(values), 4)
            if values
            else 0
        )

    return averages

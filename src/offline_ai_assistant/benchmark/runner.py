
"""Execute benchmark tests against locally installed Ollama models."""

from datetime import datetime, timezone

from offline_ai_assistant.benchmark.dataset import get_benchmark_dataset
from offline_ai_assistant.benchmark.metrics import (
    calculate_metrics,
    calculate_average_metrics,
)
from offline_ai_assistant.benchmark.evaluator import evaluate_answer


class BenchmarkRunner:
    """Run and evaluate a benchmark dataset against an Ollama model."""

    def __init__(self, model_manager):
        self.model_manager = model_manager

    def run(
        self,
        model_name: str,
        dataset: list[dict] | None = None,
    ) -> dict:
        """Execute benchmark questions and return aggregated results."""

        self.model_manager.validate_model(model_name)

        test_cases = dataset if dataset is not None else get_benchmark_dataset()

        results = []
        errors = []

        for test_case in test_cases:
            try:
                generation = self.model_manager.generate(
                    messages=[
                        {
                            "role": "user",
                            "content": test_case["question"],
                        }
                    ],
                    model_name=model_name,
                    temperature=0,
                )

                answer = generation["response"]

                metrics = calculate_metrics(
                    generation.get("metrics", {})
                )

                evaluation = evaluate_answer(
                    answer,
                    test_case["expected_answer"],
                )

                results.append({
                    "test_id": test_case["id"],
                    "category": test_case["category"],
                    "question": test_case["question"],
                    "expected_answer": test_case["expected_answer"],
                    "generated_answer": answer,
                    "metrics": metrics,
                    "evaluation": evaluation,
                    "status": "success",
                })

            except Exception as error:
                errors.append({
                    "test_id": test_case.get("id", "unknown"),
                    "error": str(error),
                })

        successful = len(results)
        total = len(test_cases)

        scores = [
            result["evaluation"]["similarity_score"]
            for result in results
        ]

        return {
            "benchmark_id": (
                f"{model_name.replace(':', '_')}_"
                f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
            ),
            "model": model_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_tests": total,
            "successful_tests": successful,
            "failed_tests": total - successful,
            "average_similarity_score": (
                round(sum(scores) / len(scores), 2)
                if scores else 0.0
            ),
            "average_metrics": calculate_average_metrics(results),
            "results": results,
            "errors": errors,
        }


"""Save and retrieve benchmark reports."""

import json
from pathlib import Path
from datetime import datetime, timezone


class BenchmarkReport:
    """Manage benchmark report storage."""

    def __init__(self, report_dir: str = "data/benchmark_reports"):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def save(self, report: dict) -> str:
        """Save a benchmark report as JSON."""

        benchmark_id = report["benchmark_id"]
        file_path = self.report_dir / f"{benchmark_id}.json"

        with file_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, ensure_ascii=False)

        return str(file_path)

    def list_reports(self) -> list[dict]:
        """Return summaries of saved benchmark reports."""

        reports = []

        for file_path in sorted(
            self.report_dir.glob("*.json"),
            reverse=True,
        ):
            try:
                with file_path.open(encoding="utf-8") as file:
                    report = json.load(file)

                reports.append({
                    "benchmark_id": report.get("benchmark_id"),
                    "model": report.get("model"),
                    "created_at": report.get("created_at"),
                    "total_tests": report.get("total_tests", 0),
                    "average_similarity_score": report.get(
                        "average_similarity_score", 0
                    ),
                })

            except (OSError, json.JSONDecodeError):
                continue

        return reports

    def get_report(self, benchmark_id: str) -> dict | None:
        """Retrieve a report using its benchmark ID."""

        file_path = self.report_dir / f"{benchmark_id}.json"

        if not file_path.is_file():
            return None

        with file_path.open(encoding="utf-8") as file:
            return json.load(file)

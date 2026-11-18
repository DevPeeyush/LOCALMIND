
"""Benchmark API routes."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from offline_ai_assistant.benchmark.runner import BenchmarkRunner
from offline_ai_assistant.benchmark.report import BenchmarkReport

router = APIRouter(prefix="/api/v1/benchmark", tags=["Benchmark"])


class BenchmarkRequest(BaseModel):
    model_name: str = Field(min_length=1)


def get_report_manager(request: Request) -> BenchmarkReport:
    manager = getattr(request.app.state, "benchmark_report", None)

    if manager is None:
        manager = BenchmarkReport()
        request.app.state.benchmark_report = manager

    return manager


@router.post("/run")
def run_benchmark(
    payload: BenchmarkRequest,
    request: Request,
):
    model_manager = getattr(request.app.state, "model_manager", None)

    if model_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Model manager is not initialized.",
        )

    try:
        runner = BenchmarkRunner(model_manager)
        report = runner.run(payload.model_name)

        get_report_manager(request).save(report)

        return report

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/reports")
def list_benchmark_reports(request: Request):
    return get_report_manager(request).list_reports()


@router.get("/reports/{benchmark_id}")
def get_benchmark_report(
    benchmark_id: str,
    request: Request,
):
    report = get_report_manager(request).get_report(benchmark_id)

    if report is None:
        raise HTTPException(
            status_code=404,
            detail="Benchmark report not found.",
        )

    return report

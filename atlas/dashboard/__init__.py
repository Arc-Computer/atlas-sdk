"""Atlas telemetry dashboard package."""

from __future__ import annotations

from .publisher import TelemetryPublisher

__all__ = ["TelemetryPublisher", "create_dashboard_app", "run_dashboard"]


def create_dashboard_app(*args, **kwargs):
    from .server import create_dashboard_app as _create_dashboard_app

    return _create_dashboard_app(*args, **kwargs)


def run_dashboard(*args, **kwargs) -> None:
    from .server import run_dashboard as _run_dashboard

    return _run_dashboard(*args, **kwargs)

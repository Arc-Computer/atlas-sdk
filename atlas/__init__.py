"""Atlas SDK package exports."""

try:
    from atlas.core import arun, run
except ModuleNotFoundError as exc:
    def _missing(*_args, **_kwargs):
        raise RuntimeError("atlas.core dependencies are not available") from exc

    arun = _missing
    run = _missing

__all__ = ["arun", "run"]

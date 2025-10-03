"""GDPval dataset loader and reference cache."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List

import httpx
from huggingface_hub import hf_hub_download

try:
    import datasets
    _DATASETS_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    datasets = None  # type: ignore[assignment]
    _DATASETS_ERROR = exc

try:
    from docx import Document  # type: ignore[import]
    _DOCX_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    Document = None  # type: ignore[assignment]
    _DOCX_ERROR = exc

try:
    from pypdf import PdfReader  # type: ignore[import]
    _PDF_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    PdfReader = None  # type: ignore[assignment]
    _PDF_ERROR = exc


class GDPValLoaderError(RuntimeError):
    """Base error raised by the GDPval loader."""


class ReferenceNotFound(GDPValLoaderError):
    """Raised when a requested reference is missing."""


class ReferenceDownloadError(GDPValLoaderError):
    """Raised when a reference cannot be downloaded or parsed."""


CACHE_ROOT = Path(os.getenv("ATLAS_GDPVAL_CACHE", ".atlas/gdpval"))


@dataclass
class GDPValReference:
    filename: str
    url: str | None = None
    hf_uri: str | None = None
    media_type: str | None = None
    checksum: str | None = None
    cached_path: Path | None = None
    text_path: Path | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def cache(self, task_id: str, fetch: "FetchFn" | None = None) -> None:
        if self.cached_path and self.cached_path.exists():
            return
        cache_dir = CACHE_ROOT / task_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / Path(self.filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        fetch_fn = fetch or _default_fetch
        try:
            content = fetch_fn(self.url, self.hf_uri)
        except Exception as exc:  # noqa: BLE001
            url_for_error = self.url or self.hf_uri or self.filename
            raise ReferenceDownloadError(f"Unable to download {url_for_error}") from exc
        target.write_bytes(content)
        self.cached_path = target
        extractor = _get_extractor(target.suffix.lower())
        text = extractor(target)
        text_path = target.with_suffix(target.suffix + ".txt")
        text_path.write_text(text)
        self.text_path = text_path
        self.metadata.update(
            {
                "source_url": self.url,
                "source_hf_uri": self.hf_uri,
                "cached_path": str(target),
                "text_path": str(text_path),
                "media_type": self.media_type,
                "checksum": self.checksum,
            }
        )


@dataclass
class GDPValTask:
    task_id: str
    sector: str
    occupation: str
    prompt: str
    references: List[GDPValReference] = field(default_factory=list)

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "sector": self.sector,
            "occupation": self.occupation,
            "prompt": self.prompt,
            "references": [ref.metadata | {"filename": ref.filename} for ref in self.references],
        }


FetchFn = Callable[[str | None, str | None], bytes]


def load_gdpval_tasks(
    *,
    split: str = "train",
    streaming: bool = False,
    cache_references: bool = False,
    fetch: FetchFn | None = None,
) -> Iterable[GDPValTask]:
    if datasets is None:  # pragma: no cover - requires optional dependency
        raise GDPValLoaderError("datasets library is required for GDPval support") from _DATASETS_ERROR
    dataset = datasets.load_dataset("openai/gdpval", split=split, streaming=streaming)
    iterator: Iterator[Dict[str, Any]]
    if streaming:
        iterator = iter(dataset)  # type: ignore[assignment]
    else:
        iterator = (record for record in dataset)  # type: ignore[assignment]
    for record in iterator:
        references = _build_references(record)
        raw_task_id = record.get("task_id") or record.get("id")
        if raw_task_id is None:
            raise GDPValLoaderError("GDPval record is missing a task identifier")
        task = GDPValTask(
            task_id=str(raw_task_id),
            sector=str(record.get("sector", "unknown")),
            occupation=str(record.get("occupation", "unknown")),
            prompt=str(record.get("prompt") or record.get("question") or ""),
            references=references,
        )
        if cache_references:
            for reference in task.references:
                reference.cache(task.task_id, fetch=fetch)
            _write_manifest(task)
        yield task


def ensure_manifest(task: GDPValTask) -> Path:
    manifest_path = CACHE_ROOT / task.task_id / "manifest.json"
    if not manifest_path.exists():
        _write_manifest(task)
    return manifest_path


def load_manifest(task_id: str) -> Dict[str, Any]:
    manifest_path = CACHE_ROOT / task_id / "manifest.json"
    if not manifest_path.exists():
        raise ReferenceNotFound(f"No cached references for task {task_id}")
    return json.loads(manifest_path.read_text())


def _write_manifest(task: GDPValTask) -> None:
    cache_dir = CACHE_ROOT / task.task_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(task.to_manifest(), indent=2))


def _default_fetch(url: str | None, hf_uri: str | None) -> bytes:
    errors: list[Exception] = []
    if url:
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                return response.content
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
    if hf_uri:
        try:
            return _fetch_from_hf(hf_uri)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
    if errors:
        raise errors[-1]
    raise ReferenceDownloadError("No reference source provided")


def _get_extractor(extension: str):
    mapping = {
        ".pdf": _extract_pdf,
        ".docx": _extract_docx,
        ".txt": _extract_text,
        ".html": _extract_text,
        ".htm": _extract_text,
    }
    return mapping.get(extension, _extract_binary)


def _extract_pdf(path: Path) -> str:
    if PdfReader is None:  # pragma: no cover - optional dependency
        raise ReferenceDownloadError("pypdf is required to extract PDF references") from _PDF_ERROR
    try:
        reader = PdfReader(path.open("rb"))
    except Exception as exc:  # noqa: BLE001
        raise ReferenceDownloadError(f"Failed to parse PDF {path}") from exc
    parts: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n".join(parts)


def _extract_docx(path: Path) -> str:
    if Document is None:  # pragma: no cover - optional dependency
        raise ReferenceDownloadError("python-docx is required to extract DOCX references") from _DOCX_ERROR
    try:
        document = Document(path)
    except Exception as exc:  # noqa: BLE001
        return _extract_binary(path)
    return "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text)


def _extract_text(path: Path) -> str:
    try:
        return path.read_text()
    except UnicodeDecodeError:
        return path.read_bytes().decode("utf-8", errors="ignore")


def _extract_binary(path: Path) -> str:
    data = path.read_bytes()
    return data.decode("utf-8", errors="ignore")


def _fetch_from_hf(hf_uri: str) -> bytes:
    parsed = _parse_hf_uri(hf_uri)
    local_path = hf_hub_download(
        repo_id=parsed["repo_id"],
        filename=parsed["path"],
        repo_type="dataset",
        revision=parsed["revision"],
    )
    return Path(local_path).read_bytes()


def _parse_hf_uri(hf_uri: str) -> Dict[str, str]:
    prefix = "hf://datasets/"
    if hf_uri.startswith(prefix):
        remainder = hf_uri[len(prefix) :]
        if "/resolve/" not in remainder:
            raise ReferenceDownloadError(f"Malformed Hugging Face URI: {hf_uri}")
        repo_id, path_part = remainder.split("/resolve/", maxsplit=1)
        if "/" not in path_part:
            raise ReferenceDownloadError(f"Missing revision or path in Hugging Face URI: {hf_uri}")
        revision, file_path = path_part.split("/", maxsplit=1)
        return {"repo_id": repo_id, "revision": revision, "path": file_path}
    # Treat non-prefixed URI as explicit repo path
    return {"repo_id": "openai/gdpval", "revision": "main", "path": hf_uri.lstrip("/")}


def _build_references(record: Dict[str, Any]) -> List[GDPValReference]:
    files = list(record.get("reference_files") or [])
    urls = list(record.get("reference_file_urls") or [])
    hf_uris = list(record.get("reference_file_hf_uris") or [])
    media_types = list(record.get("reference_media_types") or [])
    checksums = list(record.get("reference_file_checksums") or [])
    references: List[GDPValReference] = []
    for index, filename in enumerate(files):
        resolved_url = urls[index] if index < len(urls) else None
        hf_uri = hf_uris[index] if index < len(hf_uris) else None
        media_type = media_types[index] if index < len(media_types) else None
        checksum = checksums[index] if index < len(checksums) else None
        metadata = {
            "source_url": resolved_url,
            "source_hf_uri": hf_uri,
        }
        references.append(
            GDPValReference(
                filename=str(filename),
                url=resolved_url,
                hf_uri=hf_uri,
                media_type=media_type,
                checksum=checksum,
                metadata=metadata,
            )
        )
    return references


def _hf_uri_to_url(hf_uri: str | None) -> str | None:
    if not hf_uri:
        return None
    if hf_uri.startswith("http"):
        return hf_uri
    if hf_uri.startswith("hf://"):
        # hf://datasets/<repo>/resolve/<path>
        suffix = hf_uri[len("hf://") :]
        return f"https://huggingface.co/{suffix}"
    return None

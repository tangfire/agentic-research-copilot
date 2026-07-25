from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


TEXT_EXTENSIONS = {
    ".txt",
    ".text",
    ".log",
    ".md",
    ".markdown",
    ".rst",
    ".csv",
    ".json",
    ".jsonl",
    ".xml",
    ".yaml",
    ".yml",
}
HTML_EXTENSIONS = {".html", ".htm"}
MARKDOWN_EXTENSIONS = {".md", ".markdown"}
PDF_EXTENSIONS = {".pdf"}
MARKDOWN_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class DocumentReadError(ValueError):
    """Raised when a local document cannot be converted into text segments."""


@dataclass(frozen=True)
class ParsedDocumentSegment:
    title: str
    source: str
    content: str
    url: str | None
    metadata: dict[str, object]


class DocumentReader:
    """Local file reader for the internal grounding corpus.

    The reader keeps parsing separate from vector indexing. It returns one or more
    text segments with provenance metadata; DocumentStore remains responsible for
    contextual chunking, embedding, dense/BM25 fusion, and reranking.
    """

    def read_path(
        self,
        path: str | Path,
        *,
        title: str | None = None,
        source: str | None = None,
        url: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> list[ParsedDocumentSegment]:
        file_path = Path(path).expanduser().resolve()
        if not file_path.exists():
            raise DocumentReadError(f"Document path does not exist: {file_path}")
        if file_path.is_dir():
            raise DocumentReadError(f"Document path is a directory; pass a file path: {file_path}")

        suffix = file_path.suffix.lower()
        base_metadata = {
            **(metadata or {}),
            "source_path": str(file_path),
            "file_name": file_path.name,
            "file_type": suffix.lstrip(".") or "unknown",
            "file_size_bytes": file_path.stat().st_size,
        }
        base_title = title or file_path.stem
        base_source = source or str(file_path)

        if suffix in PDF_EXTENSIONS:
            return self._read_pdf(file_path, title=base_title, source=base_source, url=url, metadata=base_metadata)
        if suffix in HTML_EXTENSIONS:
            content = _html_to_text_with_headings(_read_text_file(file_path))
            return _segments_from_content(
                title=base_title,
                source=base_source,
                url=url,
                content=content,
                metadata={**base_metadata, "reader": "html_text"},
                split_sections=True,
            )
        if suffix in TEXT_EXTENSIONS or _looks_like_text(file_path):
            content = _read_text_file(file_path)
            return _segments_from_content(
                title=base_title,
                source=base_source,
                url=url,
                content=content,
                metadata={**base_metadata, "reader": "plain_text"},
                split_sections=suffix in MARKDOWN_EXTENSIONS,
            )

        raise DocumentReadError(
            f"Unsupported document type '{suffix or 'unknown'}'. "
            "Use text/Markdown/HTML files, or install PyMuPDF for PDF ingestion."
        )

    def _read_pdf(
        self,
        file_path: Path,
        *,
        title: str,
        source: str,
        url: str | None,
        metadata: dict[str, object],
    ) -> list[ParsedDocumentSegment]:
        try:
            import fitz  # type: ignore[import-not-found]
        except Exception as exc:
            raise DocumentReadError(
                "PDF ingestion requires PyMuPDF. Install the document extra with "
                "`pip install -e .[documents]` or install `pymupdf` in this environment."
            ) from exc

        document = fitz.open(str(file_path))
        try:
            page_count = len(document)
            pymupdf_version = str(getattr(fitz, "VersionBind", "unknown"))
            segments: list[ParsedDocumentSegment] = []
            for page_index, page in enumerate(document):
                page_text, page_metadata = _extract_pdf_page_text(page)
                table_text, table_metadata = _extract_pdf_tables(page)
                page_content = "\n\n".join(part for part in [page_text, table_text] if part)
                if not page_content:
                    continue
                page_number = page_index + 1
                segments.append(
                    _segment(
                        title=f"{title} p.{page_number}" if page_count > 1 else title,
                        source=source,
                        url=url,
                        content=page_content,
                        metadata={
                            **metadata,
                            "reader": "pymupdf",
                            "segment_kind": "page",
                            "parse_strategy": "pymupdf_blocks_tables_text",
                            "pymupdf_version": pymupdf_version,
                            "page_number": page_number,
                            "page_count": page_count,
                            "segment_index": page_index,
                            **page_metadata,
                            **table_metadata,
                        },
                    )
                )
        finally:
            close = getattr(document, "close", None)
            if callable(close):
                close()

        if not segments:
            raise DocumentReadError(f"No extractable text found in PDF: {file_path}")
        return segments


def _extract_pdf_page_text(page: Any) -> tuple[str, dict[str, object]]:
    metadata: dict[str, object] = {
        "pdf_text_parse_method": "text",
        "text_block_count": 0,
        "line_count": 0,
        "heading_hints": [],
    }
    rect = getattr(page, "rect", None)
    width = getattr(rect, "width", None)
    height = getattr(rect, "height", None)
    if isinstance(width, (int, float)):
        metadata["page_width"] = round(float(width), 2)
    if isinstance(height, (int, float)):
        metadata["page_height"] = round(float(height), 2)
    rotation = getattr(page, "rotation", None)
    if isinstance(rotation, (int, float)):
        metadata["page_rotation"] = int(rotation)

    blocks = _pdf_text_blocks(page)
    if blocks:
        metadata["pdf_text_parse_method"] = "blocks"
        metadata["text_block_count"] = len(blocks)
        lines = _pdf_lines_from_blocks(blocks)
        metadata["line_count"] = len(lines)
        metadata["heading_hints"] = _infer_heading_hints(lines)
        ordered_text = "\n\n".join(_normalize_text(block["text"]) for block in blocks if block["text"].strip())
        return _normalize_text(ordered_text), metadata

    try:
        text = str(page.get_text("text") or "")
    except Exception:
        text = ""
    lines = [line for line in (_normalize_text(part) for part in text.splitlines()) if line]
    metadata["line_count"] = len(lines)
    metadata["heading_hints"] = _infer_heading_hints(lines)
    return _normalize_text(text), metadata


def _pdf_text_blocks(page: Any) -> list[dict[str, object]]:
    try:
        raw_blocks = page.get_text("blocks") or []
    except Exception:
        return []

    blocks: list[dict[str, object]] = []
    for raw_block in raw_blocks:
        if not isinstance(raw_block, (list, tuple)) or len(raw_block) < 5:
            continue
        block_type = raw_block[6] if len(raw_block) > 6 else 0
        if isinstance(block_type, int) and block_type != 0:
            continue
        text = str(raw_block[4] or "").strip()
        if not text:
            continue
        x0 = _number(raw_block[0])
        y0 = _number(raw_block[1])
        x1 = _number(raw_block[2])
        y1 = _number(raw_block[3])
        blocks.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": text})
    return sorted(blocks, key=lambda block: (float(block["y0"]), float(block["x0"])))


def _pdf_lines_from_blocks(blocks: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for block in blocks:
        lines.extend(
            line
            for line in (_normalize_text(part) for part in str(block["text"]).splitlines())
            if line
        )
    return lines


def _extract_pdf_tables(page: Any) -> tuple[str, dict[str, object]]:
    metadata: dict[str, object] = {
        "table_extraction": "pymupdf_find_tables_unavailable",
        "table_count": 0,
        "table_cell_count": 0,
        "has_tables": False,
    }
    find_tables = getattr(page, "find_tables", None)
    if not callable(find_tables):
        return "", metadata

    try:
        table_result = find_tables()
    except Exception as exc:
        return "", {**metadata, "table_extraction": "pymupdf_find_tables_failed", "table_error": str(exc)}

    raw_tables = getattr(table_result, "tables", table_result)
    if raw_tables is None:
        raw_tables = []

    table_parts: list[str] = []
    table_cell_count = 0
    for index, table in enumerate(raw_tables, start=1):
        rows = _extract_table_rows(table)
        if not rows:
            continue
        table_cell_count += sum(len(row) for row in rows)
        table_parts.append(f"Detected table {index}:\n{_format_table_rows(rows)}")

    if not table_parts:
        return "", {**metadata, "table_extraction": "pymupdf_find_tables"}

    content = "PDF table extraction:\n\n" + "\n\n".join(table_parts)
    return content, {
        "table_extraction": "pymupdf_find_tables",
        "table_count": len(table_parts),
        "table_cell_count": table_cell_count,
        "has_tables": True,
    }


def _extract_table_rows(table: Any) -> list[list[str]]:
    extract = getattr(table, "extract", None)
    if callable(extract):
        try:
            rows = extract()
        except Exception:
            return []
    elif isinstance(table, (list, tuple)):
        rows = table
    else:
        return []
    cleaned_rows: list[list[str]] = []
    for row in rows or []:
        if not isinstance(row, (list, tuple)):
            continue
        cleaned = [_normalize_text(str(cell or "")) for cell in row]
        if any(cleaned):
            cleaned_rows.append(cleaned[:8])
        if len(cleaned_rows) >= 25:
            break
    return cleaned_rows


def _format_table_rows(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded_rows = [row + [""] * (width - len(row)) for row in rows]
    header = padded_rows[0]
    separator = ["---"] * width
    body = padded_rows[1:]
    markdown_rows = [header, separator, *body]
    return "\n".join("| " + " | ".join(_escape_table_cell(cell) for cell in row) + " |" for row in markdown_rows)


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _infer_heading_hints(lines: list[str]) -> list[str]:
    hints: list[str] = []
    for line in lines[:40]:
        if not _looks_like_heading(line):
            continue
        if line not in hints:
            hints.append(line)
        if len(hints) >= 5:
            break
    return hints


def _looks_like_heading(line: str) -> bool:
    line = _normalize_text(line)
    if len(line) < 4 or len(line) > 120:
        return False
    if line.endswith((".", "。", "!", "！", "?", "？", ";", "；", ":")):
        return False
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", line)
    if re.match(r"^(\d+(\.\d+)*|[IVX]+)\s+.+", line):
        return True
    if tokens and sum(1 for token in tokens if token[:1].isupper()) >= max(1, len(tokens) // 2):
        return True
    return bool(re.search(r"[\u4e00-\u9fff]", line) and len(line) <= 40)


def _number(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _segment(
    *,
    title: str,
    source: str,
    url: str | None,
    content: str,
    metadata: dict[str, object],
) -> ParsedDocumentSegment:
    normalized = _normalize_text(content)
    if not normalized:
        raise DocumentReadError(f"No extractable text found in document: {source}")
    return ParsedDocumentSegment(
        title=title,
        source=source,
        url=url,
        content=normalized,
        metadata={**metadata, "parsed_char_count": len(normalized)},
    )


def _segments_from_content(
    *,
    title: str,
    source: str,
    url: str | None,
    content: str,
    metadata: dict[str, object],
    split_sections: bool,
) -> list[ParsedDocumentSegment]:
    sections = _split_markdown_sections(content) if split_sections else []
    if not sections:
        return [
            _segment(
                title=title,
                source=source,
                url=url,
                content=content,
                metadata={**metadata, "segment_kind": "document"},
            )
        ]
    return [
        _segment(
            title=f"{title} / {section['path_text']}",
            source=source,
            url=url,
            content=section["content"],
            metadata={
                **metadata,
                "segment_kind": "section",
                "section_heading": section["heading"],
                "section_level": section["level"],
                "section_path": section["path_text"],
                "section_path_parts": section["path"],
                "section_index": index,
                "segment_index": index,
                "section_count": len(sections),
            },
        )
        for index, section in enumerate(sections)
    ]


def _split_markdown_sections(content: str) -> list[dict[str, object]]:
    lines = content.splitlines()
    sections: list[dict[str, object]] = []
    current_heading: str | None = None
    current_level = 0
    current_path: list[str] = []
    current_lines: list[str] = []
    preamble_lines: list[str] = []
    heading_stack: list[tuple[int, str]] = []

    for line in lines:
        match = MARKDOWN_HEADING_PATTERN.match(line.strip())
        if match:
            next_heading = match.group(2).strip()
            next_level = len(match.group(1))
            if current_heading is None:
                preamble = _normalize_text("\n".join(preamble_lines))
                if preamble:
                    sections.append(
                        {
                            "heading": "Overview",
                            "level": 0,
                            "path": ["Overview"],
                            "path_text": "Overview",
                            "content": preamble,
                        }
                    )
            else:
                _append_section(sections, current_heading, current_level, current_path, current_lines)
            while heading_stack and heading_stack[-1][0] >= next_level:
                heading_stack.pop()
            heading_stack.append((next_level, next_heading))
            current_heading = next_heading
            current_level = next_level
            current_path = [heading for _, heading in heading_stack]
            current_lines = []
            continue
        if current_heading is None:
            preamble_lines.append(line)
        else:
            current_lines.append(line)

    if current_heading is not None:
        _append_section(sections, current_heading, current_level, current_path, current_lines)
    return sections


def _append_section(
    sections: list[dict[str, object]],
    heading: str,
    level: int,
    path: list[str],
    lines: list[str],
) -> None:
    body = _normalize_text("\n".join(lines))
    if not body:
        return
    path = path or [heading]
    path_text = " > ".join(path)
    sections.append(
        {
            "heading": heading,
            "level": level,
            "path": path,
            "path_text": path_text,
            "content": f"{path_text}\n\n{body}",
        }
    )


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_bytes().decode("utf-8", errors="replace")


def _strip_html(value: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(value, "html.parser")
        for node in soup(["script", "style", "noscript"]):
            node.decompose()
        return soup.get_text(separator="\n", strip=True)
    except Exception:
        value = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", value)
        value = re.sub(r"(?s)<[^>]+>", " ", value)
        return value


def _html_to_text_with_headings(value: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(value, "html.parser")
        for node in soup(["script", "style", "noscript"]):
            node.decompose()
        for level in range(1, 7):
            for heading in soup.find_all(f"h{level}"):
                heading.string = f"{'#' * level} {heading.get_text(' ', strip=True)}"
        return soup.get_text(separator="\n", strip=True)
    except Exception:
        return _strip_html(value)


def _normalize_text(value: str) -> str:
    value = value.replace("\x00", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n[ \t]+", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _looks_like_text(path: Path) -> bool:
    try:
        sample = path.read_bytes()[:2048]
    except OSError:
        return False
    if b"\x00" in sample:
        return False
    return True

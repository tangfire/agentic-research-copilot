from pathlib import Path
import sys
import types

from agentic_research_copilot.document_reader import DocumentReader


def test_document_reader_loads_markdown_with_metadata(tmp_path: Path):
    path = tmp_path / "notes.md"
    path.write_text(
        "# Notes\n\nAgentic RAG depends on parsing, chunking, and reranking.\n\n"
        "## Chunking\n\nSection paths should survive ingestion.",
        encoding="utf-8",
    )

    segments = DocumentReader().read_path(path, metadata={"kind": "demo"})

    assert len(segments) == 2
    assert segments[0].title == "notes / Notes"
    assert "Agentic RAG" in segments[0].content
    assert segments[0].metadata["reader"] == "plain_text"
    assert segments[0].metadata["segment_kind"] == "section"
    assert segments[0].metadata["section_heading"] == "Notes"
    assert segments[1].metadata["section_path"] == "Notes > Chunking"
    assert segments[1].metadata["section_path_parts"] == ["Notes", "Chunking"]
    assert segments[0].metadata["file_type"] == "md"
    assert segments[0].metadata["kind"] == "demo"


def test_document_reader_strips_html_noise(tmp_path: Path):
    path = tmp_path / "paper.html"
    path.write_text(
        "<html><head><script>ignore()</script></head><body><h1>Paper</h1><p>Useful evidence.</p></body></html>",
        encoding="utf-8",
    )

    segment = DocumentReader().read_path(path)[0]

    assert "Useful evidence" in segment.content
    assert "ignore" not in segment.content
    assert segment.metadata["reader"] == "html_text"
    assert segment.metadata["segment_kind"] == "section"
    assert segment.metadata["section_heading"] == "Paper"


def test_document_reader_splits_pdf_into_page_segments(tmp_path: Path, monkeypatch):
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-1.4 fake")

    class FakePage:
        def __init__(self, text: str) -> None:
            self.text = text

        def get_text(self, mode: str) -> str:
            assert mode == "text"
            return self.text

    class FakeDocument(list):
        def close(self) -> None:
            self.closed = True

    fake_fitz = types.ModuleType("fitz")
    fake_fitz.open = lambda _path: FakeDocument([FakePage("Page one evidence."), FakePage("Page two evidence.")])
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)

    segments = DocumentReader().read_path(path, title="Research Report")

    assert [segment.metadata["page_number"] for segment in segments] == [1, 2]
    assert segments[0].title == "Research Report p.1"
    assert segments[1].metadata["reader"] == "pymupdf"
    assert segments[0].metadata["pdf_text_parse_method"] == "text"


def test_document_reader_extracts_pdf_blocks_tables_and_layout_metadata(tmp_path: Path, monkeypatch):
    path = tmp_path / "technical-report.pdf"
    path.write_bytes(b"%PDF-1.4 fake")

    class FakeRect:
        width = 612
        height = 792

    class FakeTable:
        def extract(self):
            return [["Metric", "Value"], ["Latency", "120ms"], ["Error rate", "0.1%"]]

    class FakeTableResult:
        tables = [FakeTable()]

    class FakePage:
        rect = FakeRect()
        rotation = 0

        def get_text(self, mode: str):
            if mode == "blocks":
                return [
                    (72, 40, 300, 60, "Executive Summary\n", 0, 0),
                    (72, 90, 520, 160, "The retrieval pipeline uses parent child context and reranking.\n", 1, 0),
                ]
            if mode == "text":
                return "Fallback text should not be preferred."
            raise AssertionError(mode)

        def find_tables(self):
            return FakeTableResult()

    class FakeDocument(list):
        def close(self) -> None:
            self.closed = True

    fake_fitz = types.ModuleType("fitz")
    fake_fitz.VersionBind = "1.26.0"
    fake_fitz.open = lambda _path: FakeDocument([FakePage()])
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)

    segment = DocumentReader().read_path(path, title="Technical Report")[0]

    assert "Executive Summary" in segment.content
    assert "Fallback text" not in segment.content
    assert "Detected table 1" in segment.content
    assert "| Metric | Value |" in segment.content
    assert segment.metadata["pdf_text_parse_method"] == "blocks"
    assert segment.metadata["text_block_count"] == 2
    assert segment.metadata["table_count"] == 1
    assert segment.metadata["table_cell_count"] == 6
    assert segment.metadata["has_tables"] is True
    assert segment.metadata["page_width"] == 612
    assert segment.metadata["page_height"] == 792
    assert segment.metadata["page_rotation"] == 0
    assert segment.metadata["heading_hints"] == ["Executive Summary"]

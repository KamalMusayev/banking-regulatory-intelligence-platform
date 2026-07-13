"""
app/services/document_service.py

Service layer for Document and Metadata retrieval.
Decouples raw PDF/markdown parsing and folder scanning from router endpoints.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.schemas.document import DocumentMetadataResponse, DocumentPageResponse, ArticleInfo, DocumentHighlightResponse

logger = logging.getLogger(__name__)


def extract_page_text(markdown_content: str, page_number: int) -> Optional[str]:
    """
    Extracts text content for a specific page from a cleaned markdown file.
    Cleaned markdown files use <!-- PAGE: N --> as page boundaries.
    """
    pattern = rf"<!--\s*PAGE:\s*{page_number}\s*-->"
    matches = list(re.finditer(pattern, markdown_content, re.IGNORECASE))
    if not matches:
        return None
    
    start_pos = matches[0].end()
    
    # Find the next page boundary or end of string
    next_pattern = r"<!--\s*PAGE:\s*\d+\s*-->"
    next_matches = list(re.finditer(next_pattern, markdown_content[start_pos:], re.IGNORECASE))
    if next_matches:
        end_pos = start_pos + next_matches[0].start()
    else:
        end_pos = len(markdown_content)
        
    return markdown_content[start_pos:end_pos].strip()


class DocumentService:
    """
    Service responsible for loading document catalog metadata, slicing pages,
    and resolving highlighting targets.
    """

    def __init__(
        self,
        metadata_dir: str | Path,
        cleaned_docs_dir: str | Path,
        chunk_lookup: dict[str, dict[str, Any]],
    ) -> None:
        """
        Initialize the DocumentService by recursively scanning the metadata folder.
        """
        self.metadata_dir = Path(metadata_dir)
        self.cleaned_docs_dir = Path(cleaned_docs_dir)
        self.chunk_lookup = chunk_lookup
        
        self._doc_metadata_map: Dict[str, Dict[str, Any]] = {}
        self._doc_path_map: Dict[str, Path] = {}
        self._scan_documents()

    def _scan_documents(self) -> None:
        """
        Scans all *_metadata.json files recursively to build local maps.
        """
        if not self.metadata_dir.exists():
            logger.warning("Metadata directory %s does not exist.", self.metadata_dir)
            return

        logger.info("Scanning metadata files in %s...", self.metadata_dir)
        count = 0
        for path in self.metadata_dir.rglob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                
                doc_id = meta.get("document_id")
                if doc_id:
                    self._doc_metadata_map[doc_id] = meta
                    
                    category = meta.get("category", "")
                    processed_file = meta.get("processed_file", f"{doc_id}.md")
                    md_path = self.cleaned_docs_dir / category / processed_file
                    
                    if not md_path.exists():
                        # Try fallback filename matching doc_id directly
                        md_path = self.cleaned_docs_dir / category / f"{doc_id}.md"
                    
                    self._doc_path_map[doc_id] = md_path
                    count += 1
            except Exception as e:
                logger.warning("Failed to parse metadata file at %s: %s", path, e)
                continue
        logger.info("Successfully indexed metadata and paths for %d documents.", count)

    def get_document_metadata(self, document_id: str) -> Optional[DocumentMetadataResponse]:
        """
        Finds document metadata and compiles associated articles.
        """
        meta = self._doc_metadata_map.get(document_id)
        if not meta:
            return None

        # Build list of unique articles associated with this document from chunk_lookup
        related_articles = set()
        for chunk in self.chunk_lookup.values():
            if chunk.get("document_id") == document_id:
                art = chunk.get("article")
                if art:
                    related_articles.add(str(art))

        return DocumentMetadataResponse(
            document_id=meta.get("document_id", document_id),
            title=meta.get("title") or meta.get("document_id") or "Unknown Document",
            category=meta.get("category", "unknown"),
            total_pages=meta.get("total_pages"),
            total_chunks=meta.get("total_chunks"),
            language=meta.get("language"),
            parser=meta.get("parser"),
            publication_date=meta.get("created_at"),
            status="active",
            related_articles=sorted(list(related_articles)),
            document_metadata=meta,
        )

    def get_all_documents(self) -> List[DocumentMetadataResponse]:
        """
        Retrieve all documents indexed in the system.
        """
        results = []
        for doc_id in self._doc_metadata_map:
            meta = self.get_document_metadata(doc_id)
            if meta:
                results.append(meta)
        return results

    def get_document_page(self, document_id: str, page_number: int) -> Optional[DocumentPageResponse]:
        """
        Extracts content text and finds related articles for a given document page.
        """
        meta = self._doc_metadata_map.get(document_id)
        if not meta:
            return None

        md_path = self._doc_path_map.get(document_id)
        if not md_path or not md_path.exists():
            logger.warning("Markdown source file not found for document_id %s.", document_id)
            return None

        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error("Failed to read markdown file at %s: %s", md_path, e)
            return None

        page_text = extract_page_text(content, page_number)
        if page_text is None:
            return None

        # Find any chunks that appear on this page to identify section headings
        articles = []
        for chunk in self.chunk_lookup.values():
            if chunk.get("document_id") == document_id:
                p_start = chunk.get("page_start")
                p_end = chunk.get("page_end")
                if p_start is not None and p_end is not None:
                    if p_start <= page_number <= p_end:
                        articles.append(
                            ArticleInfo(
                                chapter=chunk.get("chapter"),
                                article=chunk.get("article"),
                                section=chunk.get("section"),
                                chunk_id=chunk.get("chunk_id") or chunk.get("id") or "",
                            )
                        )

        return DocumentPageResponse(
            document_id=document_id,
            page_number=page_number,
            page_content=page_text,
            article_information=articles,
            metadata={"source_file": meta.get("source_file")},
        )

    def get_highlight(self, document_id: str, chunk_id: str) -> Optional[DocumentHighlightResponse]:
        """
        Finds chunk details to return navigation parameters for highlighting.
        """
        chunk = self.chunk_lookup.get(chunk_id)
        if not chunk:
            return None

        # Fetch basic chunk specs
        page = chunk.get("page_start") or 1
        article = chunk.get("article")
        text = chunk.get("text") or chunk.get("content") or ""

        return DocumentHighlightResponse(
            document_id=document_id,
            page=page,
            article=article,
            chunk_id=chunk_id,
            chunk_start=None,
            chunk_end=None,
            highlighted_text=text,
            offset_status="future_enhancement",
        )

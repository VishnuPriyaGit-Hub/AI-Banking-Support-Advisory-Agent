from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib import error, request
from xml.etree import ElementTree
from zipfile import ZipFile

from pymilvus import DataType, MilvusClient

from app.core.config import DOCS_DIR, RAG_SUMMARY_PATH, get_env_value

WORD_NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
ALLOWED_DOC_NAMES = {
    "Accounts.docx",
    "Bank FAQ's.docx",
    "Deposits.docx",
    "Loan.docx",
    "Cards.docx",
}
DOC_GROUP_BY_FILE = {
    "Accounts.docx": "accounts",
    "Bank FAQ's.docx": "faq",
    "Deposits.docx": "deposits",
    "Loan.docx": "loans",
    "Cards.docx": "cards",
}


@dataclass
class ChunkRecord:
    id: str
    source_file: str
    doc_group: str
    content_type: str
    chunk_index: int
    section_title: str
    text: str


class SimpleRAGIngestor:
    def __init__(
        self,
        docs_dir: Path = DOCS_DIR,
        summary_path: Path = RAG_SUMMARY_PATH,
    ) -> None:
        self.docs_dir = docs_dir
        self.summary_path = summary_path

        self.embedding_base_url = get_env_value("EMBEDDING_BASE_URL") or get_env_value("OPENAI_BASE_URL")
        self.embedding_api_key = get_env_value("EMBEDDING_API_KEY") or get_env_value("OPENAI_API_KEY")
        self.embedding_model = get_env_value("EMBEDDING_MODEL") or "text-embedding-3-small"

        self.zilliz_uri = get_env_value("ZILLIZ_ENDPOINT")
        self.zilliz_api_key = get_env_value("ZILLIZ_API_KEY")
        self.zilliz_cluster_id = get_env_value("ZILLIZ_CLUSTER_ID") or ""
        self.collection_name = get_env_value("ZILLIZ_COLLECTION_NAME") or "banking_rag_chunks"

        if not self.embedding_base_url:
            raise ValueError("EMBEDDING_BASE_URL or OPENAI_BASE_URL is required in .env.")
        if not self.embedding_api_key:
            raise ValueError("EMBEDDING_API_KEY or OPENAI_API_KEY is required in .env.")
        if not self.zilliz_uri:
            raise ValueError("ZILLIZ_ENDPOINT is required in .env.")
        if not self.zilliz_api_key:
            raise ValueError("ZILLIZ_API_KEY is required in .env.")

    def run(
        self,
        *,
        chunk_size: int = 900,
        chunk_overlap: int = 120,
        batch_size: int = 16,
        drop_existing: bool = False,
    ) -> dict[str, object]:
        documents = self.load_documents()
        chunks = self.chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            raise ValueError("No chunks were created from the selected documents.")

        embeddings = self.embed_chunks(chunks, batch_size=batch_size)
        dimension = len(embeddings[0])

        milvus = self.create_milvus_client()
        self.prepare_collection(milvus, dimension=dimension, drop_existing=drop_existing)
        insert_count = self.insert_chunks(milvus, chunks, embeddings)

        summary = {
            "docs_dir": str(self.docs_dir),
            "selected_files": sorted(documents.keys()),
            "chunk_count": len(chunks),
            "doc_groups": sorted({chunk.doc_group for chunk in chunks}),
            "embedding_model": self.embedding_model,
            "embedding_dimension": dimension,
            "collection_name": self.collection_name,
            "zilliz_endpoint": self.zilliz_uri,
            "zilliz_cluster_id": self.zilliz_cluster_id or "not_provided",
            "inserted_rows": insert_count,
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def load_documents(self) -> dict[str, list[str]]:
        if not self.docs_dir.exists():
            raise ValueError(f"Docs directory not found: {self.docs_dir}")

        documents: dict[str, list[str]] = {}
        for path in sorted(self.docs_dir.iterdir()):
            if not path.is_file() or path.name not in ALLOWED_DOC_NAMES:
                continue
            paragraphs = self.read_docx_paragraphs(path)
            if paragraphs:
                documents[path.name] = paragraphs
        return documents

    def read_docx_paragraphs(self, path: Path) -> list[str]:
        with ZipFile(path) as archive:
            xml_bytes = archive.read("word/document.xml")
        root = ElementTree.fromstring(xml_bytes)
        body = root.find(".//w:body", WORD_NAMESPACE)
        if body is None:
            return []

        blocks: list[str] = []
        for child in body:
            if child.tag == f"{W_NS}p":
                line = self.extract_paragraph_text(child)
                if line:
                    blocks.append(line)
            elif child.tag == f"{W_NS}tbl":
                blocks.extend(self.extract_table_rows(child))
        return blocks

    def extract_paragraph_text(self, paragraph: ElementTree.Element) -> str:
        texts = [node.text for node in paragraph.findall(".//w:t", WORD_NAMESPACE) if node.text]
        return self.normalize_text("".join(texts))

    def extract_table_rows(self, table: ElementTree.Element) -> list[str]:
        rows: list[str] = []
        for row in table.findall(".//w:tr", WORD_NAMESPACE):
            cells: list[str] = []
            for cell in row.findall("./w:tc", WORD_NAMESPACE):
                cell_paragraphs = [
                    self.extract_paragraph_text(paragraph)
                    for paragraph in cell.findall(".//w:p", WORD_NAMESPACE)
                ]
                cell_text = self.normalize_text(" ".join(part for part in cell_paragraphs if part))
                if cell_text:
                    cells.append(cell_text)
            if cells:
                rows.append(" | ".join(cells))
        return rows

    def chunk_documents(
        self,
        documents: dict[str, list[str]],
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        for file_name, paragraphs in documents.items():
            doc_group = DOC_GROUP_BY_FILE.get(file_name, "general")
            if doc_group == "faq":
                chunks.extend(self.chunk_faq_document(file_name, doc_group, paragraphs))
            else:
                chunks.extend(
                    self.chunk_semantic_document(
                        file_name,
                        doc_group,
                        paragraphs,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                )
        return chunks

    def chunk_faq_document(self, file_name: str, doc_group: str, paragraphs: list[str]) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        current_question = ""
        answer_parts: list[str] = []
        chunk_index = 0

        def flush_pair() -> None:
            nonlocal chunk_index, current_question, answer_parts
            if not current_question:
                return
            answer = " ".join(answer_parts).strip()
            combined = f"Question: {current_question}\nAnswer: {answer or 'Answer not provided in the source text.'}"
            chunks.append(
                ChunkRecord(
                    id=str(uuid.uuid4()),
                    source_file=file_name,
                    doc_group=doc_group,
                    content_type="faq_pair",
                    chunk_index=chunk_index,
                    section_title=current_question,
                    text=combined,
                )
            )
            chunk_index += 1
            current_question = ""
            answer_parts = []

        for paragraph in paragraphs:
            if self.looks_like_question(paragraph):
                flush_pair()
                current_question = paragraph.strip()
            elif current_question:
                answer_parts.append(paragraph.strip())
            else:
                answer_parts.append(paragraph.strip())

        flush_pair()

        if not chunks and paragraphs:
            fallback_text = "\n".join(paragraphs)
            chunks.append(
                ChunkRecord(
                    id=str(uuid.uuid4()),
                    source_file=file_name,
                    doc_group=doc_group,
                    content_type="faq_reference",
                    chunk_index=0,
                    section_title="FAQ reference",
                    text=fallback_text[:6000],
                )
            )
        return chunks

    def chunk_semantic_document(
        self,
        file_name: str,
        doc_group: str,
        paragraphs: list[str],
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[ChunkRecord]:
        sections: list[tuple[str, list[str]]] = []
        current_title = "General"
        current_paragraphs: list[str] = []

        for paragraph in paragraphs:
            if self.looks_like_heading(paragraph):
                if current_paragraphs:
                    sections.append((current_title, current_paragraphs))
                current_title = paragraph.strip()
                current_paragraphs = []
            else:
                current_paragraphs.append(paragraph.strip())

        if current_paragraphs:
            sections.append((current_title, current_paragraphs))

        chunks: list[ChunkRecord] = []
        chunk_index = 0
        for section_title, section_paragraphs in sections:
            buffer = ""
            previous_tail = ""
            for paragraph in section_paragraphs:
                candidate = f"{buffer}\n{paragraph}".strip() if buffer else paragraph
                if len(candidate) <= chunk_size:
                    buffer = candidate
                    continue

                if buffer:
                    chunk_text = self.format_section_chunk(section_title, previous_tail, buffer)
                    chunks.append(
                        ChunkRecord(
                            id=str(uuid.uuid4()),
                            source_file=file_name,
                            doc_group=doc_group,
                            content_type="section_chunk",
                            chunk_index=chunk_index,
                            section_title=section_title,
                            text=chunk_text,
                        )
                    )
                    chunk_index += 1
                    previous_tail = buffer[-chunk_overlap:].strip()
                buffer = paragraph

            if buffer:
                chunk_text = self.format_section_chunk(section_title, previous_tail, buffer)
                chunks.append(
                    ChunkRecord(
                        id=str(uuid.uuid4()),
                        source_file=file_name,
                        doc_group=doc_group,
                        content_type="section_chunk",
                        chunk_index=chunk_index,
                        section_title=section_title,
                        text=chunk_text,
                    )
                )
                chunk_index += 1
        return chunks

    def format_section_chunk(self, section_title: str, previous_tail: str, body: str) -> str:
        body_text = body.strip()
        if previous_tail:
            body_text = f"Context overlap: {previous_tail}\n{body_text}"
        return f"Section: {section_title}\n{body_text}".strip()

    def looks_like_question(self, text: str) -> bool:
        return text.strip().endswith("?")

    def looks_like_heading(self, text: str) -> bool:
        cleaned = text.strip()
        if not cleaned:
            return False
        if cleaned.endswith(":") and len(cleaned) <= 100:
            return True
        if len(cleaned) <= 70 and cleaned.isupper():
            return True
        if len(cleaned.split()) <= 8 and cleaned == cleaned.title():
            return True
        return False

    def normalize_text(self, text: str) -> str:
        text = text.replace("\r", "\n")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def embed_chunks(self, chunks: list[ChunkRecord], *, batch_size: int) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for batch in self.batch_items(chunks, batch_size):
            embeddings.extend(self.embed_text_batch([item.text for item in batch]))
        return embeddings

    def embed_text_batch(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.embedding_base_url.rstrip('/')}/embeddings"
        payload = {
            "model": self.embedding_model,
            "input": texts,
            "encoding_format": "float",
        }
        raw_request = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.embedding_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(raw_request, timeout=60) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Embedding request failed with HTTP {exc.code}: {details}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Embedding request failed: {exc.reason}") from exc

        data = response_payload.get("data", [])
        if not data:
            raise RuntimeError("Embedding response did not include vectors.")
        return [item["embedding"] for item in data]

    def create_milvus_client(self) -> MilvusClient:
        return MilvusClient(uri=self.zilliz_uri, token=self.zilliz_api_key)

    def prepare_collection(self, client: MilvusClient, *, dimension: int, drop_existing: bool) -> None:
        if client.has_collection(self.collection_name):
            if drop_existing:
                client.drop_collection(self.collection_name)
            else:
                return

        schema = client.create_schema(auto_id=False, enable_dynamic_fields=False)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field(field_name="source_file", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="doc_group", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="content_type", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
        schema.add_field(field_name="section_title", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=16384)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vector", metric_type="COSINE", index_type="AUTOINDEX")

        client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def insert_chunks(self, client: MilvusClient, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> int:
        rows = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            rows.append(
                {
                    "id": chunk.id,
                    "source_file": chunk.source_file,
                    "doc_group": chunk.doc_group,
                    "content_type": chunk.content_type,
                    "chunk_index": chunk.chunk_index,
                    "section_title": chunk.section_title,
                    "text": chunk.text,
                    "vector": embedding,
                }
            )
        client.insert(collection_name=self.collection_name, data=rows)
        return len(rows)

    def batch_items(self, items: list[ChunkRecord], batch_size: int) -> Iterable[list[ChunkRecord]]:
        for index in range(0, len(items), batch_size):
            yield items[index : index + batch_size]

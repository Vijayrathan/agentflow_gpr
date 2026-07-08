# NumPy 2.0 compatibility: np.NINF was removed, use -np.inf instead
# Patch must be applied before any dependencies import numpy
import numpy as np
if not hasattr(np, 'NINF'):
    np.NINF = -np.inf

import re
import uuid
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional

# Disable transformers' *advisory* warnings only — specifically the
# XLMRobertaTokenizerFast "use the `__call__` method instead of encode + pad"
# advice (emitted via logger.warning_advice per BGE-M3 encode batch). That
# encode-then-pad is intentional inside FlagEmbedding's m3.py: it tokenizes
# without padding, length-sorts the batches to minimize padding, then pads — a
# speed optimization we don't control and shouldn't undo. This toggle silences
# just that advisory category while leaving genuine warnings/errors visible.
# Must be set BEFORE transformers is imported — FlagEmbedding pulls it in below.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# 1. Parsing & Chunking Libraries
#    MinerU (PDF OCR parsing) is imported lazily inside parse_and_chunk so that
#    inference-only environments (and the FastAPI startup path) don't require it.
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.tools import tool
# 2. Embedding & Reranking Libraries
from FlagEmbedding import BGEM3FlagModel, FlagReranker

# 3. Vector Database (Qdrant server — accessed over its REST/gRPC API)
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Default Qdrant server endpoint; override with QDRANT_URL (e.g. in .env / docker).
DEFAULT_QDRANT_URL = "http://localhost:6333"

# Folders whose name is not an informative topic — collapsed to this sentinel so
# they never masquerade as a real category and can be filtered/down-weighted.
_NON_TOPIC_FOLDERS = {"_to_review", "own_work_proposals"}
_MISC_TOPIC = "misc"

class GeophysicsRAG:
    def __init__(self, mode: Literal["training", "inference"] = "inference", qdrant_url: str | None = None):
        """
        Initialize RAG system with mode-specific model loading.

        Args:
            mode: "training" for indexing documents, "inference" for retrieval only
            qdrant_url: URL of the Qdrant server (defaults to $QDRANT_URL or
                        http://localhost:6333). The vector DB is a standalone
                        server accessed over its REST/gRPC API — not an embedded
                        on-disk store.
        """
        self.mode = mode
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL)
        self.collection_name = "gpr_research"

        # Connect to the Qdrant server over its API.
        self.qdrant = QdrantClient(url=self.qdrant_url)

        if mode == "training":
            print("Training mode: Loading models for indexing...")
            # Training mode: Need encoder and text splitter for indexing
            self.encoder = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
            
            # Setup Semantic Chunker
            base_embedder = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
            self.text_splitter = SemanticChunker(
                base_embedder, 
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=70 
            )
            self.reranker = None  # Not needed for training
            
        elif mode == "inference":
            print("Inference mode: Loading models for retrieval...")
            # Inference mode: Need encoder and reranker for search
            self.encoder = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
            self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
            self.text_splitter = None  # Not needed for inference
        
        self._setup_collection()

    def _setup_collection(self):
        """
        Sets up the Qdrant collection with both Dense (Semantic) and Sparse
        (Keyword) named vectors, plus payload indexes for metadata filtering.

        Only creates the collection if it doesn't exist. In inference mode a
        missing collection is a soft warning (the server may simply be empty
        before the first ingestion) — `search()` guards against that.
        """
        if not self.qdrant.collection_exists(self.collection_name):
            if self.mode == "training":
                print(f"Creating collection '{self.collection_name}'...")
                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=1024,  # BGE-M3 dense dimension
                            distance=models.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams(
                            index=models.SparseIndexParams(
                                on_disk=True,  # Store sparse index on disk for persistence
                            )
                        )
                    }
                )
                self._ensure_payload_indexes()
            else:
                # Server may be freshly started with no data yet — don't hard-fail.
                print(
                    f"⚠️  Collection '{self.collection_name}' does not exist on "
                    f"{self.qdrant_url}. Run training mode to ingest documents; "
                    "searches will return no results until then."
                )
        else:
            if self.mode == "training":
                print(f"Collection '{self.collection_name}' already exists. Adding to existing collection...")
                self._ensure_payload_indexes()
            else:
                print(f"Collection '{self.collection_name}' found. Ready for inference.")

    # Payload fields we filter/scope on. Keyword for categoricals, integer for year.
    _PAYLOAD_INDEXES = {
        "topic": models.PayloadSchemaType.KEYWORD,
        "doc_type": models.PayloadSchemaType.KEYWORD,
        "source": models.PayloadSchemaType.KEYWORD,
        "year": models.PayloadSchemaType.INTEGER,
    }

    def _ensure_payload_indexes(self):
        """
        Create payload indexes so metadata filtering (topic/doc_type/source/year)
        is fast. Idempotent — re-creating an existing index is a no-op we swallow.
        """
        for field_name, schema in self._PAYLOAD_INDEXES.items():
            try:
                self.qdrant.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception as e:
                # Already exists (or transient) — indexing is best-effort here.
                print(f"  (payload index '{field_name}' skipped: {e})")

    def parse_and_chunk(self, pdf_path: str) -> List[str]:
        """
        Step 1: Layout-Aware Parsing & Semantic Chunking (via MinerU).
        Only available in training mode.
        """
        if self.mode != "training":
            raise ValueError("parse_and_chunk is only available in training mode.")

        if self.text_splitter is None:
            raise ValueError("Text splitter not initialized. Use training mode.")

        # MinerU is a heavy, training-only dependency — import it lazily so that
        # inference / API-startup paths never require it.
        from mineru.cli.common import do_parse

        pdf_path = str(Path(pdf_path).resolve())
        pdf_stem = Path(pdf_path).stem

        # Persist parsed markdown so it's inspectable after ingestion.
        parse_output_dir = Path(__file__).parent.parent / "db" / "mineru_parsed"
        parse_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Parsing {pdf_path} with MinerU...")
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        do_parse(
            output_dir=str(parse_output_dir),
            pdf_file_names=[pdf_stem],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=["en"],
            backend="pipeline",
            parse_method="ocr",
            formula_enable=True,
            table_enable=True,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=True,
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=False,
            f_make_md_mode="mm_markdown",
        )

        md_path = parse_output_dir / pdf_stem / "ocr" / f"{pdf_stem}.md"
        full_text_markdown = md_path.read_text(encoding="utf-8")
        # Strip null bytes left by any ligature-decoding failures.
        full_text_markdown = full_text_markdown.replace("\x00", "")
        print(f"Parsed markdown saved to {md_path}")

        print("Chunking text semantically...")
        docs = self.text_splitter.create_documents([full_text_markdown])
        return [d.page_content for d in docs]

    # Extensions we ingest natively as text (no OCR): already-markdown / plain text.
    _TEXT_SUFFIXES = {".md", ".txt", ".rst"}

    def _chunk_file(self, path: Path) -> List[str]:
        """
        Dispatch a source file to the right chunker: PDFs go through MinerU OCR;
        markdown/text files are read directly and semantically chunked (no OCR).
        """
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self.parse_and_chunk(str(path))
        if suffix in self._TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="ignore").replace("\x00", "")
            if not text.strip():
                return []
            docs = self.text_splitter.create_documents([text])
            return [d.page_content for d in docs]
        return []

    def _extract_metadata(self, path: Path, root: Path) -> Dict[str, Any]:
        """
        Build per-document metadata used as the payload base for every chunk.

        - topic:    top-level category folder under `root`, numeric prefix
                    stripped (e.g. `06_soil_dielectric_models` ->
                    `soil_dielectric_models`). `_to_review` / `own_work_proposals`
                    carry no informative topic -> `misc`.
        - doc_type: `documentation` for the gprMax docs folder, else derived from
                    the file extension (`pdf` / `markdown` / `text`).
        - year / author / title: parsed from the `YYYY_Author_Title` filename
                    convention used across the corpus; `XXXX_` -> year None.
        """
        try:
            rel_parts = path.relative_to(root).parts
        except ValueError:
            rel_parts = (path.name,)
        category = rel_parts[0] if len(rel_parts) > 1 else ""

        topic = re.sub(r"^\d+_", "", category)  # strip leading "NN_"
        if not topic or topic in _NON_TOPIC_FOLDERS or category in _NON_TOPIC_FOLDERS:
            topic = _MISC_TOPIC

        suffix = path.suffix.lower()
        if topic == "gprmax_documentation":
            doc_type = "documentation"
        elif suffix == ".pdf":
            doc_type = "pdf"
        elif suffix == ".md":
            doc_type = "markdown"
        else:
            doc_type = "text"

        stem = path.stem
        year: Optional[int] = None
        author: Optional[str] = None
        title = stem
        m = re.match(r"^(\d{4}|XXXX)_([^_]+)_(.*)$", stem)
        if m:
            if m.group(1).isdigit():
                year = int(m.group(1))
            author = m.group(2)
            title = m.group(3)
        title = title.replace("_", " ").replace("-", " ").strip()

        return {
            "topic": topic,
            "doc_type": doc_type,
            "year": year,
            "author": author,
            "title": title,
            "source": path.name,
        }

    def index_all_documents(self, docs_path: str | None = None):
        """
        Recursively index every supported document under `docs_path`
        (default: repo-root `review_docs/`). PDFs are OCR-parsed via MinerU;
        markdown/text files are ingested natively. Each chunk is tagged with the
        document's metadata (topic/doc_type/year/author/title/source).
        Only available in training mode.
        """
        if self.mode != "training":
            raise ValueError("index_all_documents is only available in training mode.")

        if docs_path is None:
            root = Path(__file__).parent.parent / "review_docs"
        else:
            root = Path(docs_path)
        if not root.exists():
            raise ValueError(f"Directory {root} does not exist.")

        supported = {".pdf"} | self._TEXT_SUFFIXES
        files = [
            p for p in sorted(root.rglob("*"))
            if p.is_file()
            and p.suffix.lower() in supported
            and ".claude" not in p.parts               # stray tooling dir
            and not p.name.startswith("receipt_")      # non-technical receipts
        ]

        if not files:
            print(f"No supported documents found under {root}")
            return

        print(f"Found {len(files)} documents to index under {root}...")

        all_chunks: List[str] = []
        all_metas: List[Dict[str, Any]] = []
        for f in files:
            rel = f.relative_to(root)
            print(f"\nProcessing: {rel}")
            try:
                chunks = self._chunk_file(f)
            except Exception as e:
                print(f"  Error processing {rel}: {e}")
                continue
            meta = self._extract_metadata(f, root)
            all_chunks.extend(chunks)
            all_metas.extend([meta] * len(chunks))
            print(f"  Extracted {len(chunks)} chunks (topic={meta['topic']}, type={meta['doc_type']})")

        if all_chunks:
            print(f"\nTotal chunks to index: {len(all_chunks)}")
            self.index_documents(all_chunks, all_metas)
        else:
            print("No chunks extracted from any documents.")

    def index_documents(
        self,
        chunks: List[str],
        metadatas: List[Dict[str, Any]] | None = None,
        batch_size: int = 100,
    ):
        """
        Step 2: Embed (Dense + Sparse) and Upsert to the Qdrant server.
        Only available in training mode.

        Args:
            chunks: List of text chunks to index
            metadatas: Optional per-chunk metadata (parallel to `chunks`) merged
                into each point's payload alongside `text` and a per-source
                `chunk_index` / `total_chunks`. Defaults to no metadata.
            batch_size: Number of chunks to process at once (to avoid memory issues)
        """
        if self.mode != "training":
            raise ValueError("index_documents is only available in training mode.")

        if not chunks:
            print("No chunks to index.")
            return

        if metadatas is None:
            metadatas = [{} for _ in chunks]
        if len(metadatas) != len(chunks):
            raise ValueError("metadatas must be parallel to chunks (same length).")

        # Filter out empty or whitespace-only chunks (keeping metadata aligned).
        pairs = [
            (chunk.strip(), meta)
            for chunk, meta in zip(chunks, metadatas)
            if chunk and chunk.strip()
        ]

        if not pairs:
            print("No valid (non-empty) chunks to index after filtering.")
            return

        removed_count = len(chunks) - len(pairs)
        if removed_count > 0:
            print(f"Filtered out {removed_count} empty chunks.")

        # Assign per-source chunk_index / total_chunks for provenance.
        source_totals: Dict[str, int] = {}
        for _, meta in pairs:
            source_totals[meta.get("source", "")] = source_totals.get(meta.get("source", ""), 0) + 1
        source_seen: Dict[str, int] = {}

        ingested_at = datetime.now(timezone.utc).isoformat()
        filtered_chunks = [c for c, _ in pairs]
        filtered_metas = [m for _, m in pairs]

        print(f"Indexing {len(filtered_chunks)} chunks in batches of {batch_size}...")

        # Process in batches to avoid memory issues
        total_batches = (len(filtered_chunks) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(filtered_chunks), batch_size):
            batch_chunks = filtered_chunks[batch_idx:batch_idx + batch_size]
            batch_metas = filtered_metas[batch_idx:batch_idx + batch_size]
            current_batch_num = (batch_idx // batch_size) + 1

            print(f"Processing batch {current_batch_num}/{total_batches} ({len(batch_chunks)} chunks)...")

            try:
                # BGE-M3 Encoding
                # returns dict with 'dense_vecs', 'lexical_weights', 'colbert_vecs'
                output = self.encoder.encode(batch_chunks, return_dense=True, return_sparse=True)

                points = []
                for i, (chunk, meta) in enumerate(zip(batch_chunks, batch_metas)):
                    # Format Dense Vector
                    dense_vec = output['dense_vecs'][i]

                    # Format Sparse Vector (Convert dictionary to Qdrant format)
                    sparse_weight_dict = output['lexical_weights'][i]
                    sparse_indices = [int(k) for k in sparse_weight_dict.keys()]
                    sparse_values = list(sparse_weight_dict.values())

                    src = meta.get("source", "")
                    chunk_index = source_seen.get(src, 0)
                    source_seen[src] = chunk_index + 1

                    payload = {
                        **meta,
                        "text": chunk,
                        "chunk_index": chunk_index,
                        "total_chunks": source_totals.get(src, 1),
                        "ingested_at": ingested_at,
                    }

                    points.append(models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "dense": dense_vec,
                            "sparse": models.SparseVector(indices=sparse_indices, values=sparse_values)
                        },
                        payload=payload
                    ))

                # Upsert batch to Qdrant
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"  ✓ Batch {current_batch_num} indexed successfully.")

            except Exception as e:
                print(f"  ✗ Error processing batch {current_batch_num}: {e}")
                print(f"    Skipping {len(batch_chunks)} chunks from this batch.")
                continue

        print(f"Indexing complete. Processed {len(filtered_chunks)} chunks.")

    def search(self, query: str, top_k: int = 5, query_filter: "models.Filter | None" = None):
        """
        Step 3: Hybrid Retrieval + Reranking.
        Only available in inference mode.

        Uses a single server-side hybrid query: dense + sparse prefetch fused
        with Reciprocal Rank Fusion (RRF) on the Qdrant server, then a
        cross-encoder rerank pass. An optional `query_filter` (Qdrant `Filter`)
        scopes retrieval by metadata (e.g. exclude `topic="misc"`, or restrict to
        `doc_type="documentation"`).
        """
        if self.mode != "inference":
            raise ValueError("search is only available in inference mode.")

        if self.reranker is None:
            raise ValueError("Reranker not initialized. Use inference mode.")

        # Guard: server may have no collection yet (freshly started, pre-ingest).
        if not self.qdrant.collection_exists(self.collection_name):
            print(f"⚠️  Collection '{self.collection_name}' not found on {self.qdrant_url}.")
            print("   Please run in training mode first to index documents.")
            return []

        collection_info = self.qdrant.get_collection(self.collection_name)
        if collection_info.points_count == 0:
            print(f"⚠️  Warning: Collection '{self.collection_name}' is empty!")
            print("   Please run in training mode first to index documents.")
            return []

        print(f"Searching for: '{query}' (Collection has {collection_info.points_count} documents)")

        # 1. Encode Query — wrap in list so BGE-M3 always returns batch format
        q_output = self.encoder.encode([query], return_dense=True, return_sparse=True)

        q_dense = q_output['dense_vecs'][0]
        q_sparse_weights = q_output['lexical_weights'][0]

        q_sparse_indices = [int(k) for k in q_sparse_weights.keys()]
        q_sparse_values = list(q_sparse_weights.values())

        # 2. Hybrid Search — server-side RRF fusion of dense + sparse prefetches.
        try:
            response = self.qdrant.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(query=q_dense, using="dense", limit=20, filter=query_filter),
                    models.Prefetch(
                        query=models.SparseVector(indices=q_sparse_indices, values=q_sparse_values),
                        using="sparse",
                        limit=20,
                        filter=query_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=20,
                with_payload=True,
            )
            points = response.points
        except Exception as e:
            print(f"⚠️  Error during hybrid search: {e}")
            print("Trying fallback dense search...")
            fallback_response = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=q_dense,
                using="dense",
                limit=20,
                with_payload=True,
                query_filter=query_filter,
            )
            points = fallback_response.points

        # Extract documents from results
        candidate_docs = [
            point.payload["text"]
            for point in points
            if getattr(point, "payload", None) and "text" in point.payload
        ]

        if not candidate_docs:
            print("⚠️  No candidate documents found. The query might not match any indexed content.")
            return []

        print(f"Found {len(candidate_docs)} candidate documents for reranking...")
        
        # 3. Reranking (The "Geophysics Judge")
        # Rerank the top 20 candidates to find the true best matches
        print("Reranking candidates...")
        pairs = [[query, doc] for doc in candidate_docs]
        scores = self.reranker.compute_score(pairs, normalize=True)
        
        # Handle both single score and list of scores
        if not isinstance(scores, list):
            scores = [scores]
        
        # Combine docs with scores and sort
        ranked_results = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
        
        return ranked_results[:top_k]
    
    def generate(self, query: str, top_k: int = 3) -> str:
        """
        Step 4: Generate answer using LLM with retrieved context.
        
        Args:
            query: User's question
            top_k: Number of documents to retrieve for context
            
        Returns:
            Generated answer as a string
        """
        from langchain_openai import ChatOpenAI
        import os
        
        # Get retrieved documents
        results = self.search(query, top_k=top_k)
        
        if not results:
            return (
                "I couldn't find specific information about that in the knowledge base."
            )
        
        # Prepare context from retrieved documents
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            context_parts.append(f"[Source {i}] (Relevance: {score:.2f})\n{doc}\n")
        
        context = "\n".join(context_parts)
        
        # Create prompt for LLM
        system_prompt = """You are a helpful assistant specializing in Ground Penetrating Radar (GPR) and geophysics.
Your role is to answer questions based on the provided research documents and technical documentation.

Guidelines:
1. Answer the user's question directly and concisely
2. Use information from the provided sources
3. If the sources mention specific values, equations, or technical details, include them
4. If the answer involves multiple sources, synthesize the information
5. Be technical but clear
6. If the sources don't fully answer the question, say so and provide what information is available
7. At the end, offer to help with creating a GPR simulation if relevant"""

        user_prompt = f"""Based on the following research documents, please answer this question:

Question: {query}

Retrieved Documents:
{context}

Please provide a clear, informative answer based on these sources."""

        # Initialize LLM
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        llm = ChatOpenAI(model=openai_model, api_key=openai_api_key, temperature=0.3)
        
        # Generate response
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = llm.invoke(messages)
        
        return response.content
    
    def list_sources(self) -> Dict[str, int]:
        """
        Maintenance helper: enumerate indexed documents by source filename with
        their chunk counts, by scrolling the collection's payloads. Maps to
        Qdrant's scroll (`GET`) API.
        """
        if not self.qdrant.collection_exists(self.collection_name):
            return {}
        counts: Dict[str, int] = {}
        next_page = None
        while True:
            points, next_page = self.qdrant.scroll(
                collection_name=self.collection_name,
                with_payload=["source"],
                with_vectors=False,
                limit=1000,
                offset=next_page,
            )
            for p in points:
                src = (p.payload or {}).get("source", "<unknown>")
                counts[src] = counts.get(src, 0) + 1
            if next_page is None:
                break
        return counts

    def delete_by_source(self, source: str) -> None:
        """
        Maintenance helper: delete every chunk belonging to a source filename,
        via a Qdrant filtered delete (`DELETE` API). Idempotent.
        """
        self.qdrant.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(
                        key="source", match=models.MatchValue(value=source)
                    )]
                )
            ),
        )

    def close(self):
        """
        Properly close the Qdrant client to avoid cleanup warnings.
        """
        if hasattr(self, 'qdrant') and self.qdrant is not None:
            try:
                self.qdrant.close()
            except Exception:
                pass  # Ignore errors during cleanup
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()

_rag_instance: GeophysicsRAG | None = None


def init_rag() -> GeophysicsRAG:
    """
    Eagerly build the inference RAG system (loads the BGE-M3 encoder + reranker
    and connects to the Qdrant server) and cache it as the process singleton.

    Intended to be called once from the FastAPI startup/lifespan hook so the RAG
    is "ready to serve" before the first request, instead of loading lazily on
    the first `rag_search` call. Idempotent — returns the existing instance if
    already initialized.
    """
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = GeophysicsRAG(mode="inference")
    return _rag_instance


def _get_rag() -> GeophysicsRAG:
    """Return the cached RAG system, lazily building it if startup init was
    skipped (e.g. in tests). Prefer `init_rag()` at server startup."""
    return init_rag()


RELEVANCE_THRESHOLD = 0.7


@tool
def rag_search(
    query: Annotated[str, "The question to search the geophysics knowledge base for."],
) -> str:
    """Search the geophysics knowledge base (research papers, GPR docs, soil
    property references). Returns relevant passages above the relevance
    threshold, or 'NO_RESULTS' if nothing relevant is found."""
    import traceback
    try:
        rag_system = _get_rag()
        results = rag_system.search(query, top_k=3)
        if not results:
            return "NO_RESULTS"
        # Filter by relevance threshold
        relevant = [(doc, score) for doc, score in results if score >= RELEVANCE_THRESHOLD]
        if not relevant:
            return "NO_RESULTS"
        parts = []
        for i, (doc, score) in enumerate(relevant, 1):
            parts.append(f"[Passage {i}, relevance={score:.3f}]\n{doc}")
            print(f"Relevance: {score:.3f}")
            print(doc[:100] + "...")
            print("-" * 60)
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"RAG_SEARCH ERROR:\n{tb}")
        return f"RAG_SEARCH_ERROR: {e}\n{tb}"




# --- Usage Example ---
if __name__ == "__main__":
    import sys
    
    # Determine mode from command line or default to inference
    mode = sys.argv[1] if len(sys.argv) > 1 else "inference"
    
    if mode == "training":
        # Training mode: Index all documents from retrieval_docs
        print("=" * 60)
        print("TRAINING MODE: Indexing documents")
        print("=" * 60)
        rag = None
        try:
            rag = GeophysicsRAG(mode="training")
            rag.index_all_documents()
            print("\n✓ Training complete! Documents indexed and stored.")
            print("  You can now use inference mode to search.")
        finally:
            if rag is not None:
                rag.close()
        
    elif mode == "inference":
        # Inference mode: Only perform retrieval
        print("=" * 60)
        print("INFERENCE MODE: Retrieval only")
        print("=" * 60)
        rag = None
        try:
            rag = GeophysicsRAG(mode="inference")
            
            # Example search
            query = "What is the frequency range of peplinski model?"
            results = rag.search(query)
            
            print("\n--- Final Results ---")
            for i, (doc, score) in enumerate(results, 1):
                print(f"\n[Result {i}, Score: {score:.4f}]")
                print(f"{doc[:200]}..." if len(doc) > 200 else doc)
        finally:
            if rag is not None:
                rag.close()
    else:
        print(f"Invalid mode: {mode}. Use 'training' or 'inference'")
        print("\nUsage:")
        print("  python rag.py training   # Index documents")
        print("  python rag.py inference  # Search documents (default)")
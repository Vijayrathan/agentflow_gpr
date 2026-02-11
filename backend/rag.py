# NumPy 2.0 compatibility: np.NINF was removed, use -np.inf instead
# Patch must be applied before any dependencies import numpy
import numpy as np
if not hasattr(np, 'NINF'):
    np.NINF = -np.inf

import uuid
import os
from pathlib import Path
from typing import List, Dict, Any, Literal

# 1. Parsing & Chunking Libraries
from docling.document_converter import DocumentConverter  # Layout-aware parser
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# 2. Embedding & Reranking Libraries
from FlagEmbedding import BGEM3FlagModel, FlagReranker

# 3. Vector Database
from qdrant_client import QdrantClient
from qdrant_client.http import models

class GeophysicsRAG:
    def __init__(self, mode: Literal["training", "inference"] = "inference", qdrant_path: str = "./qdrant_storage"):
        """
        Initialize RAG system with mode-specific model loading.
        
        Args:
            mode: "training" for indexing documents, "inference" for retrieval only
            qdrant_path: Path to store Qdrant database on disk
        """
        self.mode = mode
        self.qdrant_path = qdrant_path
        self.collection_name = "gpr_research"
        
        # Create Qdrant storage directory if it doesn't exist
        os.makedirs(qdrant_path, exist_ok=True)
        
        # Initialize Vector DB (Qdrant) - Persistent storage on disk
        self.qdrant = QdrantClient(path=qdrant_path)
        
        if mode == "training":
            print("Training mode: Loading models for indexing...")
            # Training mode: Need encoder and text splitter for indexing
            self.encoder = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
            
            # Setup Semantic Chunker
            base_embedder = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
            self.text_splitter = SemanticChunker(
                base_embedder, 
                breakpoint_threshold_type="percentile" # Splits when topics shift significantly
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
        Sets up Qdrant with support for both Dense (Semantic) and Sparse (Keyword) vectors.
        Only creates collection if it doesn't exist.
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
            else:
                # In inference mode, collection should already exist
                raise ValueError(f"Collection '{self.collection_name}' does not exist. Please run in training mode first to index documents.")
        else:
            if self.mode == "training":
                print(f"Collection '{self.collection_name}' already exists. Adding to existing collection...")
            else:
                print(f"Collection '{self.collection_name}' found. Ready for inference.")

    def parse_and_chunk(self, pdf_path: str) -> List[str]:
        """
        Step 1: Layout-Aware Parsing & Semantic Chunking
        Only available in training mode.
        """
        if self.mode != "training":
            raise ValueError("parse_and_chunk is only available in training mode.")
        
        if self.text_splitter is None:
            raise ValueError("Text splitter not initialized. Use training mode.")
        
        print(f"Parsing {pdf_path}...")
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        
        # Export to Markdown to preserve headers/tables
        full_text_markdown = result.document.export_to_markdown()
        
        print("Chunking text semantically...")
        docs = self.text_splitter.create_documents([full_text_markdown])
        return [d.page_content for d in docs]
    
    def index_all_documents(self, retrieval_docs_path: str = "../retrieval_docs"):
        """
        Index all PDF documents from the retrieval_docs directory.
        Only available in training mode.
        """
        if self.mode != "training":
            raise ValueError("index_all_documents is only available in training mode.")
        
        retrieval_path = Path(retrieval_docs_path)
        if not retrieval_path.exists():
            raise ValueError(f"Directory {retrieval_docs_path} does not exist.")
        
        # Find all PDF files
        pdf_files = list(retrieval_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {retrieval_docs_path}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to index...")
        
        all_chunks = []
        for pdf_file in pdf_files:
            print(f"\nProcessing: {pdf_file.name}")
            try:
                chunks = self.parse_and_chunk(str(pdf_file))
                all_chunks.extend(chunks)
                print(f"  Extracted {len(chunks)} chunks from {pdf_file.name}")
            except Exception as e:
                print(f"  Error processing {pdf_file.name}: {e}")
                continue
        
        if all_chunks:
            print(f"\nTotal chunks to index: {len(all_chunks)}")
            self.index_documents(all_chunks)
        else:
            print("No chunks extracted from any documents.")

    def index_documents(self, chunks: List[str], batch_size: int = 100):
        """
        Step 2: Embed (Dense + Sparse) and Upsert to DB
        Only available in training mode.
        
        Args:
            chunks: List of text chunks to index
            batch_size: Number of chunks to process at once (to avoid memory issues)
        """
        if self.mode != "training":
            raise ValueError("index_documents is only available in training mode.")
        
        if not chunks:
            print("No chunks to index.")
            return
        
        # Filter out empty or whitespace-only chunks
        filtered_chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
        
        if not filtered_chunks:
            print("No valid (non-empty) chunks to index after filtering.")
            return
        
        removed_count = len(chunks) - len(filtered_chunks)
        if removed_count > 0:
            print(f"Filtered out {removed_count} empty chunks.")
        
        print(f"Indexing {len(filtered_chunks)} chunks in batches of {batch_size}...")
        
        # Process in batches to avoid memory issues
        total_batches = (len(filtered_chunks) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(filtered_chunks), batch_size):
            batch_chunks = filtered_chunks[batch_idx:batch_idx + batch_size]
            current_batch_num = (batch_idx // batch_size) + 1
            
            print(f"Processing batch {current_batch_num}/{total_batches} ({len(batch_chunks)} chunks)...")
            
            try:
                # BGE-M3 Encoding
                # returns dict with 'dense_vecs', 'lexical_weights', 'colbert_vecs'
                output = self.encoder.encode(batch_chunks, return_dense=True, return_sparse=True)
                
                points = []
                for i, chunk in enumerate(batch_chunks):
                    # Format Dense Vector
                    dense_vec = output['dense_vecs'][i]
                    
                    # Format Sparse Vector (Convert dictionary to Qdrant format)
                    sparse_weight_dict = output['lexical_weights'][i]
                    sparse_indices = [int(k) for k in sparse_weight_dict.keys()]
                    sparse_values = list(sparse_weight_dict.values())
                    
                    points.append(models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "dense": dense_vec,
                            "sparse": models.SparseVector(indices=sparse_indices, values=sparse_values)
                        },
                        payload={"text": chunk}
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

    def search(self, query: str, top_k: int = 5):
        """
        Step 3: Hybrid Retrieval + Reranking
        Only available in inference mode.
        """
        if self.mode != "inference":
            raise ValueError("search is only available in inference mode.")
        
        if self.reranker is None:
            raise ValueError("Reranker not initialized. Use inference mode.")
        
        # Check if collection has any documents
        collection_info = self.qdrant.get_collection(self.collection_name)
        if collection_info.points_count == 0:
            print(f"⚠️  Warning: Collection '{self.collection_name}' is empty!")
            print("   Please run in training mode first to index documents.")
            return []
        
        print(f"Searching for: '{query}' (Collection has {collection_info.points_count} documents)")
        
        # 1. Encode Query
        q_output = self.encoder.encode(query, return_dense=True, return_sparse=True)
        
        # Handle both single query and batch encoding
        if isinstance(q_output['dense_vecs'], list):
            q_dense = q_output['dense_vecs'][0] if len(q_output['dense_vecs']) > 0 else q_output['dense_vecs']
        else:
            q_dense = q_output['dense_vecs']
        
        # Convert query sparse weights to Qdrant format
        q_sparse_weights = q_output['lexical_weights']
        if isinstance(q_sparse_weights, list):
            q_sparse_weights = q_sparse_weights[0] if len(q_sparse_weights) > 0 else {}
        
        q_sparse_indices = [int(k) for k in q_sparse_weights.keys()]
        q_sparse_values = list(q_sparse_weights.values())

        # 2. Hybrid Search (Fusion of Dense + Sparse)
        # Search using both dense and sparse vectors, then combine results
        candidate_ids = set()
        candidate_map = {}
        
        try:
            # Dense vector search
            dense_response = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=q_dense,
                using="dense",
                limit=20,
                with_payload=True
            )
            
            # Extract points from response
            dense_results = dense_response.points if hasattr(dense_response, 'points') else dense_response
            
            for result in dense_results:
                point_id = result.id if hasattr(result, 'id') else str(result)
                candidate_ids.add(point_id)
                if point_id not in candidate_map:
                    candidate_map[point_id] = result
            
            # Sparse vector search
            sparse_response = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=models.SparseVector(indices=q_sparse_indices, values=q_sparse_values),
                using="sparse",
                limit=20,
                with_payload=True
            )
            
            # Extract points from response
            sparse_results = sparse_response.points if hasattr(sparse_response, 'points') else sparse_response
            
            for result in sparse_results:
                point_id = result.id if hasattr(result, 'id') else str(result)
                candidate_ids.add(point_id)
                if point_id not in candidate_map:
                    candidate_map[point_id] = result
            
            # Combine results (simple union for now, could use RRF scoring)
            points = list(candidate_map.values())
            
        except Exception as e:
            print(f"⚠️  Error during hybrid search: {e}")
            print("Trying fallback dense search...")
            # Fallback to simple dense search
            fallback_response = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=q_dense,
                using="dense",
                limit=20,
                with_payload=True
            )
            points = fallback_response.points if hasattr(fallback_response, 'points') else fallback_response
        
        # Extract documents from results
        candidate_docs = []
        for point in points:
            if hasattr(point, 'payload') and point.payload and 'text' in point.payload:
                candidate_docs.append(point.payload['text'])
            elif isinstance(point, dict) and 'payload' in point and 'text' in point['payload']:
                candidate_docs.append(point['payload']['text'])
        
        if not candidate_docs:
            print("⚠️  No candidate documents found. The query might not match any indexed content.")
            return []

        print(f"Found {len(candidate_docs)} candidate documents for reranking...")
        
        # 3. Reranking (The "Geophysics Judge")
        # Rerank the top 20 candidates to find the true best matches
        print("Reranking candidates...")
        pairs = [[query, doc] for doc in candidate_docs]
        scores = self.reranker.compute_score(pairs)
        
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
                "I couldn't find specific information about that in the knowledge base. "
                "However, I can help you create a GPR simulation. "
                "Would you like to start setting up a simulation instead?"
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
            rag.index_all_documents("../retrieval_docs")
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
import logging
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class ChromaDBManager:
    """
    Manager for interacting with ChromaDB for storing and retrieving embeddings.
    """
    
    def __init__(self, persist_directory: str = "data/chroma"):
        """
        Initialize the ChromaDB manager.
        
        Args:
            persist_directory: Path to store ChromaDB data persistently.
        """
        logger.info(f"Initializing ChromaDB manager at {persist_directory}")
        self.client = chromadb.PersistentClient(path=persist_directory)

    def get_or_create_collection(self, collection_name: str):
        """
        Get a collection by name or create it if it doesn't exist.
        """
        logger.info(f"Getting or creating collection: {collection_name}")
        # Using l2 distance or cosine depending on the model, default is l2 in Chroma.
        # We can specify cosine space if preferred, but for normalized vectors l2 is equivalent to cosine in ranking.
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def insert_chunks(self, collection_name: str, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Insert document chunks and their embeddings into a collection.
        
        Args:
            collection_name: The name of the collection to insert into.
            chunks: A list of chunk dictionaries containing at least 'id', 'text', and metadata.
            embeddings: A list of embedding vectors corresponding to the chunks.
        """
        if not chunks or not embeddings:
            logger.warning("Empty chunks or embeddings list provided.")
            return

        if len(chunks) != len(embeddings):
            raise ValueError(f"Number of chunks ({len(chunks)}) does not match number of embeddings ({len(embeddings)}).")

        collection = self.get_or_create_collection(collection_name)
        
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            # We assume chunk has 'chunk_id' or 'id'. Fallback to string index if not found (though it should be).
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if not chunk_id:
                raise ValueError("Chunk must contain a 'chunk_id' or 'id' field.")
                
            ids.append(str(chunk_id))
            
            # Text content of the chunk
            text = chunk.get("text") or chunk.get("content", "")
            documents.append(text)
            
            # Prepare metadata, extracting only valid metadata types (str, int, float, bool)
            # ChromaDB doesn't support nested dicts or lists in metadata directly
            meta = {}
            for k, v in chunk.items():
                if k not in ["text", "content"]: # Exclude the main text
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                    elif v is None:
                        meta[k] = ""
                    else:
                        meta[k] = str(v)
            metadatas.append(meta)

        logger.info(f"Inserting {len(ids)} chunks into collection {collection_name}")
        
        # Batch insert into ChromaDB
        # ChromaDB recommended batch size is usually < 41666, we are likely well within this.
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Successfully inserted {len(ids)} chunks.")

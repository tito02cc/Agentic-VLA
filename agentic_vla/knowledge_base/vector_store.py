"""Core module implementation."""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np


@dataclass
class VectorStoreConfig:
    """VectorStoreConfig class."""
    persist_directory: str = "./data/chroma_db"
    collection_name: str = "grasp_experiences"


class VectorStore(ABC):


    """VectorStore class."""
    @abstractmethod
    def add(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """add function."""
        pass
        
    @abstractmethod
    def query_by_text(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """query_by_text function."""
        pass
        
    @abstractmethod
    def query_by_category(
        self,
        category: str,
    ) -> List[Dict[str, Any]]:
        """query_by_category function."""
        pass
        
    @abstractmethod
    def get_visual_embedding(
        self,
        experience_id: str,
    ) -> Optional[np.ndarray]:
        """get_visual_embedding function."""
        pass
        
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """delete function."""
        pass
        
    @abstractmethod
    def count(self) -> int:
        """count function."""
        pass


class ChromaVectorStore(VectorStore):


    """ChromaVectorStore class."""
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        self.config = config or VectorStoreConfig()
        self.client = None
        self.collection = None
        self._initialized = False
        
    def initialize(self) -> None:
        
        """initialize function."""
        if self._initialized:
            return
            
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Note

            os.makedirs(self.config.persist_directory, exist_ok=True)
            
            # Note

            self.client = chromadb.PersistentClient(
                path=self.config.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            
            # Note

            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"},  # Note
            )
            
            print(f"[ChromaVectorStore] Initialized at {self.config.persist_directory}")
            print(f"[ChromaVectorStore] Collection '{self.config.collection_name}' has {self.collection.count()} entries")
            
            self._initialized = True
            
        except ImportError:
            raise ImportError("chromadb is required: pip install chromadb")
            
    def add(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]],
        documents: Optional[List[str]] = None,
    ) -> None:
        """add function."""
        if not self._initialized:
            self.initialize()
            
        # Note

        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
            
        # Note

        processed_metadatas = []
        for meta in metadatas:
            processed = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    processed[k] = v
                elif isinstance(v, dict):
                    # Note

                    import json
                    processed[k] = json.dumps(v, ensure_ascii=False)
                elif isinstance(v, list):
                    processed[k] = json.dumps(v, ensure_ascii=False)
                else:
                    processed[k] = str(v)
            processed_metadatas.append(processed)
            
        # Note

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=processed_metadatas,
            documents=documents or [""] * len(ids),
        )
        
        print(f"[ChromaVectorStore] Added {len(ids)} entries")
        
    def query_by_text(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """query_by_text function."""
        if not self._initialized:
            self.initialize()
            
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
            
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["metadatas", "distances", "embeddings"],
        )
        
        # Note

        output = []
        if results["ids"] and results["ids"][0]:
            for i, exp_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                
                # Note

                import json
                for k, v in metadata.items():
                    if isinstance(v, str) and v.startswith(("{", "[")):
                        try: metadata[k] = json.loads(v)
                        except:
                            pass
                            
                output.append({
                    "id": exp_id,
                    "score": 1 - distance,  # Note
                    **metadata,
                })
                
        return output
        
    def query_by_category(
        self,
        category: str,
    ) -> List[Dict[str, Any]]:
        """query_by_category function."""
        if not self._initialized:
            self.initialize()
            
        results = self.collection.get(
            where={"category": category},
            include=["metadatas", "embeddings"],
        )
        
        output = []
        if results["ids"]:
            import json
            for i, exp_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                
                # Note

                for k, v in metadata.items():
                    if isinstance(v, str) and v.startswith(("{", "[")):
                        try: metadata[k] = json.loads(v)
                        except:
                            pass
                            
                output.append({
                    "id": exp_id,
                    **metadata,
                })
                
        return output
        
    def get_visual_embedding(
        self,
        experience_id: str,
    ) -> Optional[np.ndarray]:
        """get_visual_embedding function."""
        if not self._initialized:
            self.initialize()
            
        results = self.collection.get(
            ids=[experience_id],
            include=["metadatas"],
        )
        
        if results["metadatas"] and results["metadatas"][0]:
            meta = results["metadatas"][0]
            if "visual_embedding" in meta:
                import json
                emb = meta["visual_embedding"]
                if isinstance(emb, str):
                    emb = json.loads(emb)
                return np.array(emb)
                
        return None
        
    def delete(self, ids: List[str]) -> None:
        
        """delete function."""
        if not self._initialized:
            self.initialize()
            
        self.collection.delete(ids=ids)
        print(f"[ChromaVectorStore] Deleted {len(ids)} entries")
        
    def count(self) -> int:
        
        """count function."""
        if not self._initialized:
            self.initialize()
        return self.collection.count()
        
    def clear(self) -> None:
        
        """clear function."""
        if not self._initialized:
            self.initialize()
            
        # Note

        self.client.delete_collection(self.config.collection_name)
        self.collection = self.client.create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[ChromaVectorStore] Store reset complete")
        
    def get_all(self) -> List[Dict[str, Any]]:
        
        """get_all function."""
        if not self._initialized:
            self.initialize()
            
        results = self.collection.get(include=["metadatas"])
        
        output = []
        if results["ids"]:
            import json
            for i, exp_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                for k, v in metadata.items():
                    if isinstance(v, str) and v.startswith(("{", "[")):
                        try: metadata[k] = json.loads(v)
                        except:
                            pass
                output.append({"id": exp_id, **metadata})
                
        return output

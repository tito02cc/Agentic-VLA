"""Core module implementation."""

import os
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import OrderedDict
import numpy as np
from PIL import Image

from .schema import GraspExperience, GraspPose, ObjectInfo, Position3D, Quaternion
from .vector_store import ChromaVectorStore, VectorStoreConfig


class ShortTermMemory:


    """ShortTermMemory class."""
    def __init__(self, max_size: int = 50, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple] = OrderedDict()  # id -> (experience, timestamp)
        
    def add(self, experience: GraspExperience) -> None:
        
        """add function."""
        # Note

        if experience.id in self._cache:
            self._cache.move_to_end(experience.id)
        else:
            # Note

            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
                
        self._cache[experience.id] = (experience, datetime.now())
        
    def get(self, experience_id: str) -> Optional[GraspExperience]:
        
        """get function."""
        self._cleanup_expired()
        
        if experience_id in self._cache:
            exp, _ = self._cache[experience_id]
            self._cache.move_to_end(experience_id)  # Note
            return exp
        return None
        
    def get_recent(self, n: int = 10) -> List[GraspExperience]:
        
        """get_recent function."""
        self._cleanup_expired()
        
        items = list(self._cache.values())[-n:]
        return [exp for exp, _ in items]
        
    def get_by_category(self, category: str) -> List[GraspExperience]:
        
        """get_by_category function."""
        self._cleanup_expired()
        
        return [
            exp for exp, _ in self._cache.values()
            if exp.object_info.category == category
        ]
        
    def _cleanup_expired(self) -> None:
        
        """_cleanup_expired function."""
        now = datetime.now()
        expired = [
            k for k, (_, ts) in self._cache.items()
            if (now - ts).total_seconds() > self.ttl_seconds
        ]
        for k in expired:
            del self._cache[k]
            
    def clear(self) -> None:
            
        """clear function."""
        self._cache.clear()
        
    def __len__(self) -> int:
        return len(self._cache)


class GraspMemory:


    """GraspMemory class."""
    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        collection_name: str = "grasp_experiences",
        short_term_size: int = 50,
        short_term_ttl: int = 3600,
    ):
        # Note

        self.vector_store = ChromaVectorStore(VectorStoreConfig(
            persist_directory=persist_dir,
            collection_name=collection_name,
        ))
        
        # Note

        self.short_term = ShortTermMemory(short_term_size, short_term_ttl)
        
        # Note

        self._embedding_model = None
        
    @property
    def embedding_model(self):
        """embedding_model function."""
        if self._embedding_model is None:
            from ..core.embedding import get_embedding_model
            self._embedding_model = get_embedding_model()
        return self._embedding_model
        
    def add_experience(
        self,
        object_info: ObjectInfo,
        scene_image: Image.Image,
        grasp_pose: GraspPose,
        success: bool = True,
        confidence: float = 1.0,
        save_image: bool = True,
        image_dir: str = "./data/grasp_images",
        to_long_term: bool = True,
    ) -> GraspExperience:
        """add_experience function."""
        # Note

        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Note

        image_path = ""
        if save_image:
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f"{exp_id}.jpg")
            scene_image.save(image_path, "JPEG", quality=90)
            
        # Note

        experience = GraspExperience(
            id=exp_id,
            object_info=object_info,
            scene_image_path=image_path,
            grasp_pose=grasp_pose,
            success=success,
            confidence=confidence,
        )
        
        # Note

        self.short_term.add(experience)
        
        # Note

        if to_long_term and success:
            # Note
            self._add_to_vector_store(experience, scene_image)
            
        return experience
        
    def _add_to_vector_store(
        self,
        experience: GraspExperience,
        image: Image.Image,
    ) -> None:
        """_add_to_vector_store function."""
        # Note

        text_desc = f"{experience.object_info.category} {experience.object_info.name} {experience.object_info.description}"
        text_embedding = self.embedding_model.encode_text(text_desc)[0]
        visual_embedding = self.embedding_model.encode_image(image)[0]
        
        # Note

        metadata = experience.to_knowledge_entry()
        metadata["visual_embedding"] = visual_embedding.tolist()
        
        # Note

        self.vector_store.add(
            ids=[experience.id],
            embeddings=text_embedding.reshape(1, -1),  # Note
            metadatas=[metadata],
            documents=[text_desc],
        )
        
    def query(
        self,
        query_image: Optional[Image.Image] = None,
        query_text: Optional[str] = None,
        category: Optional[str] = None,
        n_results: int = 10,
        include_short_term: bool = True,
    ) -> List[Dict[str, Any]]:
        """query function."""
        results = []
        
        # Note

        if include_short_term:
            if category:
                short_term_results = self.short_term.get_by_category(category)
            else:
                short_term_results = self.short_term.get_recent(n_results)
                
            for exp in short_term_results:
                results.append({
                    "source": "short_term",
                    **exp.to_knowledge_entry(),
                })
                
        # Note

        if query_text:
            query_embedding = self.embedding_model.encode_text(query_text)[0]
            
            where = {"category": category} if category else None
            long_term_results = self.vector_store.query_by_text(
                query_embedding=query_embedding,
                n_results=n_results,
                where=where,
            )
            
            for item in long_term_results:
                item["source"] = "long_term"
                results.append(item)
                
        elif category:
            # Note

            long_term_results = self.vector_store.query_by_category(category)
            for item in long_term_results[:n_results]:
                item["source"] = "long_term"
                results.append(item)
                
        return results
        
    def get_experience(self, experience_id: str) -> Optional[GraspExperience]:
        
        """get_experience function."""
        # Note

        exp = self.short_term.get(experience_id)
        if exp:
            return exp
            
        # Note

        results = self.vector_store.query_by_category("")  # Note
        for item in results:
            if item["id"] == experience_id:
                return GraspExperience.from_knowledge_entry(item)
                
        return None
        
    def get_statistics(self) -> Dict[str, Any]:
        
        """get_statistics function."""
        return {
            "short_term_count": len(self.short_term),
            "long_term_count": self.vector_store.count(),
        }
        
    def export_to_json(self, filepath: str) -> None:
        
        """export_to_json function."""
        all_experiences = self.vector_store.get_all()
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(all_experiences, f, ensure_ascii=False, indent=2)
            
        print(f"[GraspMemory] Exported {len(all_experiences)} experiences to {filepath}")
        
    def import_from_json(self, filepath: str) -> int:
        
        """import_from_json function."""
        with open(filepath, "r", encoding="utf-8") as f:
            experiences = json.load(f)
            
        count = 0
        for exp_data in experiences:
            try:
                # Note

                text_desc = f"{exp_data.get('category', '')} {exp_data.get('object_name', '')} {exp_data.get('description', '')}"
                text_embedding = self.embedding_model.encode_text(text_desc)[0]
                
                self.vector_store.add(
                    ids=[exp_data["id"]],
                    embeddings=text_embedding.reshape(1, -1),
                    metadatas=[exp_data],
                    documents=[text_desc],
                )
                count += 1
            except Exception as e:
                print(f"[GraspMemory] Operation failed: {e}")
                
        print(f"[GraspMemory] Loaded {count} experiences")
        return count


def create_sample_experience(
    name: str = "test_object",
    category: str = "cup",
    description: str = "A sample test object",
) -> GraspExperience:
    """create_sample_experience function."""
    return GraspExperience(
        id=f"sample_{uuid.uuid4().hex[:8]}",
        object_info=ObjectInfo(
            name=name,
            category=category,
            description=description,
        ),
        scene_image_path="",
        grasp_pose=GraspPose(
            position=Position3D(x=0.3, y=0.1, z=0.15),
            orientation=Quaternion.identity(),
            gripper_width=0.06,
        ),
        success=True,
        confidence=0.95,
    )

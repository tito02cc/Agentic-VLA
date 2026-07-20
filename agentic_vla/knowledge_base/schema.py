"""Core module implementation."""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
import numpy as np


class Position3D(BaseModel):


    """Position3D class."""
    x: float = Field(..., description="X coordinate (meters)")
    y: float = Field(..., description="Y coordinate (meters)")
    z: float = Field(..., description="Z coordinate (meters)")
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
        
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Position3D":
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))


class Quaternion(BaseModel):


    """Quaternion class."""
    qx: float = Field(0.0, description="X component")
    qy: float = Field(0.0, description="Y component")
    qz: float = Field(0.0, description="Z component")
    qw: float = Field(1.0, description="W component (scalar)")
    
    def to_array(self) -> np.ndarray:
        return np.array([self.qx, self.qy, self.qz, self.qw])
        
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Quaternion":
        return cls(qx=float(arr[0]), qy=float(arr[1]), qz=float(arr[2]), qw=float(arr[3]))
        
    @classmethod
    def identity(cls) -> "Quaternion":
        return cls(qx=0.0, qy=0.0, qz=0.0, qw=1.0)


class GraspPose(BaseModel):


    """GraspPose class."""
    position: Position3D = Field(..., description="Grasp target position")
    orientation: Quaternion = Field(
        default_factory=Quaternion.identity,
        description="Grasp orientation (quaternion)"
    )
    gripper_width: float = Field(0.08, description="Gripper opening width (meters)")
    approach_vector: Optional[Position3D] = Field(
        None, description="Embedding vector"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": {"x": self.position.x, "y": self.position.y, "z": self.position.z},
            "orientation": {
                "qx": self.orientation.qx,
                "qy": self.orientation.qy,
                "qz": self.orientation.qz,
                "qw": self.orientation.qw,
            },
            "gripper_width": self.gripper_width,
            "approach_vector": self.approach_vector.model_dump() if self.approach_vector else None,
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraspPose":
        return cls(
            position=Position3D(**data["position"]),
            orientation=Quaternion(**data.get("orientation", {})),
            gripper_width=data.get("gripper_width", 0.08),
            approach_vector=Position3D(**data["approach_vector"]) if data.get("approach_vector") else None,
        )


class BoundingBox(BaseModel):


    """BoundingBox class."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    
    @property
    def center(self) -> tuple:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)
        
    @property
    def width(self) -> float:
        return self.x_max - self.x_min
        
    @property
    def height(self) -> float:
        return self.y_max - self.y_min


class ObjectInfo(BaseModel):


    """ObjectInfo class."""
    name: str = Field(..., description="Object name")
    category: str = Field(..., description="Object category (e.g. cup, tool, container)")
    description: str = Field("", description="Object description")
    
    # Note

    material: Optional[str] = Field(None, description="Material type (e.g. glass, metal, plastic)")
    weight_kg: Optional[float] = Field(None, description="Weight in kg")
    fragile: bool = Field(False, description="Whether the object is fragile")
    
    # Note

    bounding_box: Optional[BoundingBox] = Field(None, description="2D bounding box")
    segmentation_mask: Optional[str] = Field(None, description="File path")
    
    # Note

    visual_embedding: Optional[List[float]] = Field(None, description="Embedding vector")
    text_embedding: Optional[List[float]] = Field(None, description="Embedding vector")


class GraspExperience(BaseModel):


    """GraspExperience class."""
    id: str = Field(..., description="Unique experience ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")
    
    # Note

    object_info: ObjectInfo = Field(..., description="Target object information")
    
    # Note

    scene_image_path: str = Field(..., description="Scene image file path")
    object_crop_path: Optional[str] = Field(None, description="Object crop image file path")
    depth_image_path: Optional[str] = Field(None, description="Depth image file path")
    
    # Note

    grasp_pose: GraspPose = Field(..., description="6-DOF grasp pose")
    
    # Note

    success: bool = Field(True, description="Whether the grasp was successful")
    confidence: float = Field(1.0, description="Confidence score (0-1)")
    
    # Note

    gripper_type: str = Field("parallel_jaw", description="Gripper type")
    task_description: Optional[str] = Field(None, description="Task description")
    
    # Note

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def to_knowledge_entry(self) -> Dict[str, Any]:
    
        """to_knowledge_entry function."""
        return {
            "id": self.id,
            "object_name": self.object_info.name,
            "category": self.object_info.category,
            "description": self.object_info.description,
            "image_path": self.scene_image_path,
            "grasp_pose": self.grasp_pose.to_dict(),
            "success": self.success,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
        
    @classmethod
    def from_knowledge_entry(cls, entry: Dict[str, Any]) -> "GraspExperience":
        """from_knowledge_entry function."""
        return cls(
            id=entry["id"],
            timestamp=datetime.fromisoformat(entry.get("timestamp", datetime.now().isoformat())),
            object_info=ObjectInfo(
                name=entry["object_name"],
                category=entry["category"],
                description=entry.get("description", ""),
            ),
            scene_image_path=entry["image_path"],
            grasp_pose=GraspPose.from_dict(entry["grasp_pose"]),
            success=entry.get("success", True),
            confidence=entry.get("confidence", 1.0),
            metadata=entry.get("metadata", {}),
        )


class GraspPrediction(BaseModel):


    """GraspPrediction class."""
    grasp_pose: GraspPose = Field(..., description="The 6D grasp pose")
    confidence: float = Field(..., description="confidence 0-1")
    reasoning: str = Field("", description="Reasoning explanation")
    
    # Note

    retrieved_experiences: List[str] = Field(
        default_factory=list,
        description="experiencesIDEN"
    )
    
    # Note

    inference_time_ms: float = Field(0.0, description="inference_time")
    retrieval_time_ms: float = Field(0.0, description="retrieval_time")

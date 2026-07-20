import json
import logging
import os
from typing import List, Dict, Any, Optional
try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LifelongMemoryManager")

class LifelongMemoryManager:
    """
    基于 RESEARCH_DIRECTIONS.md 优化的: 结构化知识表示与终身学习 (Memory Consolidation)
    引入 ChromaDB 实现真正的经验存储与权重衰减
    """
    def __init__(self, persist_dir: str = "agentic_vla/knowledge_base/memory_store"):
        self.persist_dir = persist_dir
        self.conflict_threshold = 3 
        
        # 初始化 ChromaDB 向量数据库用于持久化记忆
        if HAS_CHROMA:
            # 确保持久化目录存在
            os.makedirs(self.persist_dir, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
            
            # 使用一个 Dummy Embedding Function 避免在国内下载 ONNX all-MiniLM-L6-v2 模型超时
            from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
            class DummyEmbeddingFunction(EmbeddingFunction):
                def __call__(self, input: Documents) -> Embeddings:
                    return [[0.0] * 384 for _ in input]
                    
            self.collection = self.chroma_client.get_or_create_collection(
                name="grasp_affordance_memory", 
                embedding_function=DummyEmbeddingFunction()
            )
            logger.info("ChromaDB Client Initialized for Lifelong Memory (with Dummy Embedder).")
        else:
            self.chroma_client = None
            logger.warning("ChromaDB not installed, falling back to in-memory store.")
            
        # 内部缓存 (Buffer) 或 Fallback
        self.memory_store: Dict[str, Dict] = {
            "cup": {"success_count": 10, "fail_count": 2, "z_offset": 0.05, "grasp_point": "rim", "force": "soft_grip"},
            "mustard": {"success_count": 15, "fail_count": 0, "z_offset": 0.01, "grasp_point": "body_center", "force": "high_grip"},
            "sugar": {"success_count": 5, "fail_count": 5, "z_offset": 0.00, "grasp_point": "side_edges", "force": "soft_grip"},
        }

    def _sync_to_chroma(self, obj_key: str, data: Dict):
        """将结构化的命题式和处方式知识同步至 ChromaDB"""
        if self.chroma_client is not None:
            doc_str = f"Object: {obj_key}, strategy: pull from {data['grasp_point']}, z_offset: {data['z_offset']}, force: {data['force']}"
            metadata = data.copy()
            metadata["object"] = obj_key
            metadata["is_unknown"] = False
            # Upsert
            self.collection.upsert(
                documents=[doc_str],
                metadatas=[metadata],
                ids=[f"mem_{obj_key}"]
            )

    def extract_prior(self, object_name: str) -> Dict:
        """从存储中获取先验环境事实与处方式知识"""
        for k, v in self.memory_store.items():
            if k in object_name.lower():
                return v
        return {"success_count": 0, "fail_count": 0, "z_offset": 0.015, "grasp_point": "center", "force": "medium_grip", "is_unknown": True}

    def consolidate_memory(self, object_name: str, execution_result: Dict) -> None:
        """经验提炼：成功增加权重，失败累积并在阈值时衰减/更新"""
        obj_key = next((k for k in self.memory_store.keys() if k in object_name.lower()), None)
        
        # 如果是没见过的物体，直接追加进入知识库 (提炼)
        if obj_key is None:
            obj_key = object_name.lower()
            self.memory_store[obj_key] = {"success_count": 0, "fail_count": 0, "z_offset": 0.015, "grasp_point": "center", "force": "medium_grip"}

        if execution_result.get("status") == "success":
            self.memory_store[obj_key]["success_count"] += 1
            if HAS_CHROMA: self._sync_to_chroma(obj_key, self.memory_store[obj_key])
            logger.info(f"Memory Consolidated: Increased confidence for {obj_key}.")
        else:
            self.memory_store[obj_key]["fail_count"] += 1
            logger.warning(f"Negative Reward Logged: {obj_key} failed. Total fails: {self.memory_store[obj_key]['fail_count']}")
            
            # 记忆衰减机制：如果特定静态知识频繁触发物理反馈失败
            if self.memory_store[obj_key]["fail_count"] >= self.conflict_threshold:
                logger.error(f"Memory Conflict Threshold Reached for {obj_key}!")
                self._apply_memory_decay_and_vlm_update(obj_key, execution_result)
                if HAS_CHROMA: self._sync_to_chroma(obj_key, self.memory_store[obj_key])

    def _apply_memory_decay_and_vlm_update(self, obj_key: str, execution_result: Dict) -> None:
        """冲突与衰减：重新评估先验知识，更新 Z 偏移量或力控力度"""
        logger.info(f"VLM Autonomous Reflecting on {obj_key}...")
        # 简化 VLM 开源词汇反思。我们把失败的归因假定为物理错位
        if "timeout" in execution_result.get("message", "").lower() or "collision" in execution_result.get("message", "").lower():
            # 物理下压碰到了阻力，提高 z 轴
            old_z = self.memory_store[obj_key]['z_offset']
            self.memory_store[obj_key]['z_offset'] = old_z + 0.02
            logger.info(f"Knowledge Update: {obj_key} tends to collide. Adjusted z_offset {old_z} -> {self.memory_store[obj_key]['z_offset']}")
        else:
            # 可能是滑脱，增加力度
            self.memory_store[obj_key]['force'] = "high_grip"
            logger.info(f"Knowledge Update: {obj_key} tends to slip. Adjusted force to high_grip.")
            
        # 衰减失败计数器，给予重新信用的机会
        self.memory_store[obj_key]["fail_count"] = 0
        self.memory_store[obj_key]["success_count"] = int(self.memory_store[obj_key]["success_count"] * 0.5) # 权重衰减

class ContextManager:
    """
    基于 RESEARCH_DIRECTIONS.md 优化的: 上下文管理系统 (Context Caching)
    增加对短时动作执行序列的分层缓存与逐渐压缩机制，包含 VLM 视觉历史（如 SnapKV 逻辑）
    """
    def __init__(self, max_visual_frames=3):
        self.short_term_buffer = []
        self.visual_frame_cache = [] # 存储图像帧的缓存
        self.max_visual_frames = max_visual_frames

    def log_action(self, action_name: str, status: str, details: str, visual_frame=None):
        self.short_term_buffer.append({"action": action_name, "status": status, "details": details})
        
        if visual_frame is not None:
            self.visual_frame_cache.append(visual_frame)
            # 动态舍弃冗余帧 (类似 SnapKV 粗略截断机制)
            if len(self.visual_frame_cache) > self.max_visual_frames:
                # 仅保留第一帧(全局信息)和最后几帧 (当前关键动作)
                self.visual_frame_cache = [self.visual_frame_cache[0]] + self.visual_frame_cache[-(self.max_visual_frames-1):]

    def progressive_compression(self) -> str:
        """渐进式上下文压缩: 对短时动作执行序列，清理冗余帧，仅保留工具调用骨架和状态结论"""
        if len(self.short_term_buffer) > 5:
            success_count = sum(1 for a in self.short_term_buffer if a["status"] == "success")
            fail_count = sum(1 for a in self.short_term_buffer if a["status"] != "success")
            
            # 清空冗长队列，将其压成一句精炼总结
            summary = f"Summary: Executed {len(self.short_term_buffer)} atomic actions. {success_count} successes, {fail_count} failures."
            
            # 只保留最后的关键失败点（如果有）用于 VLM 分析
            last_failed = next((a for a in reversed(self.short_term_buffer) if a["status"] != "success"), None)
            
            self.short_term_buffer.clear()
            self.short_term_buffer.append({"action": "COMPRESSED_MICRO_CONTEXT", "status": "success", "details": summary})
            
            if last_failed:
                self.short_term_buffer.append(last_failed)
                
            return summary
        return "Context ok, no compression needed."

    def get_state(self):
        return self.short_term_buffer

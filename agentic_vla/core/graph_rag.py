import networkx as nx
import numpy as np
import logging

logger = logging.getLogger("GraphRAG")

class GraphRAGManager:
    """
    基于 RESEARCH_DIRECTIONS.md 优化的: 基于 Graph RAG 的复杂场景空间关联检索
    """
    def __init__(self):
        self.graph_db = [] # List of tuples: (graph: nx.DiGraph, strategy: dict)

    def build_scene_graph(self, actors, target_actor):
        """将目标周围的局部场景映射为多模态图结构"""
        G = nx.DiGraph()
        if target_actor is None:
            return G
            
        try:
            if hasattr(target_actor, 'pose'):
                target_obj_pose = target_actor.pose.p
                if hasattr(target_obj_pose, 'cpu'):
                    target_pos = target_obj_pose[0].cpu().numpy()
                else:
                    target_pos = target_obj_pose[0] if len(target_obj_pose.shape) > 1 else target_obj_pose
            else:
                # Robosuite objects don't have pose directly in this format
                return G
        except Exception:
            return G
        G.add_node("target", cls=target_actor.name.split('_')[-1])

        for a in actors:
            if a == target_actor:
                continue
            name = a.name.lower()
            if any(ignore in name for ignore in ["table", "ground", "goal", "agent"]): 
                continue
            
            # Simple spatial relationship
            try:
                if hasattr(a, 'pose'):
                    other_obj_pose = a.pose.p
                    if hasattr(other_obj_pose, 'cpu'):
                        other_pos = other_obj_pose[0].cpu().numpy()
                    else:
                        other_pos = other_obj_pose[0] if len(other_obj_pose.shape) > 1 else other_obj_pose
                else:
                    continue
            except Exception:
                continue
                
            dist = np.linalg.norm(target_pos - other_pos)
            if dist < 0.15: # in vicinity (15cm)
                dz = other_pos[2] - target_pos[2]
                node_name = f"context_{a.name.split('_')[-1]}"
                G.add_node(node_name, cls=a.name.split('_')[-1])
                
                # Check occlusion or support
                if dz > 0.05:
                    G.add_edge("target", node_name, relation="occluded_by")
                elif dz < -0.05:
                    G.add_edge(node_name, "target", relation="supports")
                else:
                    G.add_edge("target", node_name, relation="next_to")
                    
        return G

    def retrieve_strategy(self, current_graph):
        """遇到复杂堆叠场景时，匹配数据库中类似子图拓扑结构的被遮挡/支撑抓取策略"""
        logger.info("🕸️ [Graph RAG] Executing topology-aware subgraph matching...")
        
        best_match_score = 0
        best_strategy = None
        
        for saved_graph, strategy in self.graph_db:
             # Basic Graph Edit Distance or isomorphic check substitute
             if nx.is_isomorphic(current_graph, saved_graph):
                 return strategy
             
             # Fallback simple matching score based on edges
             common_edges = len(set(current_graph.edges) & set(saved_graph.edges))
             if common_edges > best_match_score:
                 best_match_score = common_edges
                 best_strategy = strategy
        
        if best_strategy:
             logger.info(f"    -> [Graph RAG] Subgraph match found! Score: {best_match_score}. Adjusting trajectory to avoid occlusions.")
             return best_strategy
             
        logger.info("    -> [Graph RAG] No complex topology match found. Reverting to basic Target-Centric prior.")
        return None

    def store_graph_experience(self, current_graph, applied_strategy):
        """动态追加包含拓扑关联边的经验到 ChromaDB（这里用内存表代替）"""
        self.graph_db.append((current_graph, applied_strategy))
        logger.info("🕸️ [Graph RAG] Successfully committed scene topology graph into Spatial Memory!")

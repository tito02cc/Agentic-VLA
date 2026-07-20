from .core.memory_and_context import LifelongMemoryManager, ContextManager
from .agent.optimized_multi_agent import MultiAgentOrchestrator

import mani_skill.envs
import gymnasium as gym
import numpy as np

def run_experiment():
    print("=============================================================")
    print("🚀 [Agentic-RAG-VLM] Orchestrator System Booting...")
    print("=============================================================")
    env = gym.make(
        "PickClutterYCB-v1",
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        render_mode="human",
        num_envs=1
    )
    env.reset(seed=42)
    env.render()
    
    # 实例化基于 RESEARCH_DIRECTIONS 的新架构核心群
    orchestrator = MultiAgentOrchestrator(env)
    
    # 1. 场景提取 (Vision)
    entities = orchestrator.vision_agent()
    storage_bin = np.array([-0.25, 0.4, 0.25])
    
    # 2. 从上到下编排执行 (Execution & Critic in DAG loop)
    try:
        orchestrator.execution_agent(entities, storage_bin)
    except KeyboardInterrupt:
        print("\n⏹️ User interrupted the simulation pipeline.")
        
    env.close()

if __name__ == "__main__":
    run_experiment()

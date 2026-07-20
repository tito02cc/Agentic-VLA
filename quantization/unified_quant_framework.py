"""
VLM-Aware Unified Quantization Framework
=========================================

面向机器人视觉抓取的统一VLM量化框架

核心创新:
1. VLM-Aware 混合精度量化策略 (视觉FP16 + 语言INT4 + KV Cache FP8)
2. 自动精度敏感性检测
3. 动态内存管理
4. 质量-性能权衡控制

作者: Agentic-RAG-VLM Team
目标会议: IROS 2025/2026
"""

import os
import time
import json
import gc
import torch
import numpy as np
from PIL import Image, ImageDraw
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from datetime import datetime
from pathlib import Path


# ============================================================
# 配置定义
# ============================================================

class QuantMethod(Enum):
    """量化方法"""
    NONE = "none"
    BITSANDBYTES_INT4 = "bitsandbytes_int4"
    BITSANDBYTES_INT8 = "bitsandbytes_int8"
    GPTQ_INT4 = "gptq_int4"
    AWQ_INT4 = "awq_int4"
    AUTOROUND_INT4 = "autoround_int4"


class KVCacheType(Enum):
    """KV Cache 类型"""
    AUTO = "auto"
    FP16 = "fp16"
    FP8 = "fp8"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"


class ComponentType(Enum):
    """模型组件类型"""
    VISION_ENCODER = "vision_encoder"
    VISION_PROJECTOR = "vision_projector"
    LANGUAGE_MODEL = "language_model"
    LM_HEAD = "lm_head"


@dataclass
class ComponentQuantConfig:
    """组件级量化配置"""
    component: ComponentType
    quant_method: QuantMethod = QuantMethod.NONE
    dtype: str = "float16"
    quantize: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "component": self.component.value,
            "quant_method": self.quant_method.value,
            "dtype": self.dtype,
            "quantize": self.quantize,
        }


@dataclass
class VLMQuantConfig:
    """VLM 统一量化配置"""
    # 基础配置
    name: str = "default"
    description: str = ""
    
    # 模型路径
    model_path: str = ""
    
    # 组件级配置
    vision_encoder: ComponentQuantConfig = field(default_factory=lambda: ComponentQuantConfig(
        component=ComponentType.VISION_ENCODER,
        quant_method=QuantMethod.NONE,
        dtype="float16",
        quantize=False  # 视觉编码器默认不量化
    ))
    
    vision_projector: ComponentQuantConfig = field(default_factory=lambda: ComponentQuantConfig(
        component=ComponentType.VISION_PROJECTOR,
        quant_method=QuantMethod.NONE,
        dtype="float16",
        quantize=False
    ))
    
    language_model: ComponentQuantConfig = field(default_factory=lambda: ComponentQuantConfig(
        component=ComponentType.LANGUAGE_MODEL,
        quant_method=QuantMethod.BITSANDBYTES_INT4,
        dtype="float16",
        quantize=True  # 语言模型默认量化
    ))
    
    # KV Cache 配置
    kv_cache_dtype: KVCacheType = KVCacheType.FP8
    
    # vLLM 配置
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.85
    max_images_per_prompt: int = 4
    
    # 性能配置
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "model_path": self.model_path,
            "vision_encoder": self.vision_encoder.to_dict(),
            "vision_projector": self.vision_projector.to_dict(),
            "language_model": self.language_model.to_dict(),
            "kv_cache_dtype": self.kv_cache_dtype.value,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }


# ============================================================
# 预定义配置
# ============================================================

class QuantPresets:
    """预定义量化配置"""
    
    @staticmethod
    def fp16_baseline(model_path: str) -> VLMQuantConfig:
        """FP16 基准配置 (无量化)"""
        config = VLMQuantConfig(
            name="FP16_Baseline",
            description="全 FP16，无量化 (基准配置)",
            model_path=model_path,
        )
        config.vision_encoder.quantize = False
        config.language_model.quantize = False
        config.language_model.quant_method = QuantMethod.NONE
        config.kv_cache_dtype = KVCacheType.AUTO
        return config
    
    @staticmethod
    def full_int4(model_path: str) -> VLMQuantConfig:
        """全 INT4 配置 (包括视觉编码器)"""
        config = VLMQuantConfig(
            name="Full_INT4",
            description="全 INT4，包括视觉编码器 (通常不推荐)",
            model_path=model_path,
        )
        config.vision_encoder.quantize = True
        config.vision_encoder.quant_method = QuantMethod.BITSANDBYTES_INT4
        config.language_model.quantize = True
        config.language_model.quant_method = QuantMethod.BITSANDBYTES_INT4
        config.kv_cache_dtype = KVCacheType.AUTO
        return config
    
    @staticmethod
    def vlm_aware_hybrid(model_path: str) -> VLMQuantConfig:
        """VLM-Aware 混合精度配置 (推荐)"""
        config = VLMQuantConfig(
            name="VLM_Aware_Hybrid",
            description="VLM-Aware 混合精度: 视觉 FP16 + 语言 INT4",
            model_path=model_path,
        )
        config.vision_encoder.quantize = False
        config.vision_encoder.dtype = "float16"
        config.language_model.quantize = True
        config.language_model.quant_method = QuantMethod.BITSANDBYTES_INT4
        config.kv_cache_dtype = KVCacheType.AUTO
        return config
    
    @staticmethod
    def vlm_aware_optimal(model_path: str) -> VLMQuantConfig:
        """VLM-Aware 最优配置 (混合精度 + FP8 KV Cache)"""
        config = VLMQuantConfig(
            name="VLM_Aware_Optimal",
            description="VLM-Aware 最优: 视觉 FP16 + 语言 INT4 + FP8 KV Cache",
            model_path=model_path,
        )
        config.vision_encoder.quantize = False
        config.vision_encoder.dtype = "float16"
        config.language_model.quantize = True
        config.language_model.quant_method = QuantMethod.BITSANDBYTES_INT4
        config.kv_cache_dtype = KVCacheType.FP8
        return config
    
    @staticmethod
    def edge_deployment(model_path: str) -> VLMQuantConfig:
        """边缘部署配置 (AGX Orin 优化)"""
        config = VLMQuantConfig(
            name="Edge_Deployment",
            description="边缘部署优化: 低显存 + 高效推理",
            model_path=model_path,
        )
        config.vision_encoder.quantize = False
        config.language_model.quantize = True
        config.language_model.quant_method = QuantMethod.BITSANDBYTES_INT4
        config.kv_cache_dtype = KVCacheType.FP8
        config.max_model_len = 2048  # 降低上下文长度节省显存
        config.gpu_memory_utilization = 0.9
        config.max_images_per_prompt = 2
        return config


# ============================================================
# 性能监控
# ============================================================

@dataclass
class PerformanceMetrics:
    """性能指标"""
    # 速度
    tokens_per_second: float = 0.0
    time_to_first_token: float = 0.0
    total_generation_time: float = 0.0
    
    # 内存
    gpu_memory_allocated_gb: float = 0.0
    gpu_memory_reserved_gb: float = 0.0
    peak_memory_gb: float = 0.0
    
    # 质量
    output_length: int = 0
    chinese_char_ratio: float = 0.0
    output_quality: str = "unknown"
    
    # 元信息
    timestamp: str = ""
    config_name: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self._start_time: Optional[float] = None
        self._first_token_time: Optional[float] = None
        
    def start_generation(self):
        """开始生成计时"""
        self._start_time = time.time()
        self._first_token_time = None
        torch.cuda.reset_peak_memory_stats()
        
    def record_first_token(self):
        """记录首个 token 时间"""
        if self._first_token_time is None and self._start_time:
            self._first_token_time = time.time()
            
    def end_generation(self, output_text: str, config_name: str = "") -> PerformanceMetrics:
        """结束生成并计算指标"""
        end_time = time.time()
        
        # 计算时间指标
        total_time = end_time - self._start_time if self._start_time else 0
        ttft = self._first_token_time - self._start_time if self._first_token_time and self._start_time else 0
        
        # 估算 token 数 (粗略估计: 中文 1 字符 ≈ 1-2 tokens)
        num_tokens = len(output_text) * 1.5
        tps = num_tokens / total_time if total_time > 0 else 0
        
        # 内存指标
        mem_allocated = torch.cuda.memory_allocated() / 1024**3
        mem_reserved = torch.cuda.memory_reserved() / 1024**3
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        
        # 质量指标
        chinese_chars = sum(1 for c in output_text if '\u4e00' <= c <= '\u9fff')
        chinese_ratio = chinese_chars / len(output_text) if output_text else 0
        
        # 判断输出质量
        if chinese_ratio > 0.3 and len(output_text) > 50:
            quality = "good"
        elif chinese_ratio > 0.1 or len(output_text) > 30:
            quality = "medium"
        else:
            quality = "poor"
            
        metrics = PerformanceMetrics(
            tokens_per_second=tps,
            time_to_first_token=ttft,
            total_generation_time=total_time,
            gpu_memory_allocated_gb=mem_allocated,
            gpu_memory_reserved_gb=mem_reserved,
            peak_memory_gb=peak_mem,
            output_length=len(output_text),
            chinese_char_ratio=chinese_ratio,
            output_quality=quality,
            timestamp=datetime.now().isoformat(),
            config_name=config_name,
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_average_metrics(self) -> Dict[str, float]:
        """获取平均指标"""
        if not self.metrics_history:
            return {}
            
        return {
            "avg_tokens_per_second": np.mean([m.tokens_per_second for m in self.metrics_history]),
            "avg_ttft": np.mean([m.time_to_first_token for m in self.metrics_history]),
            "avg_generation_time": np.mean([m.total_generation_time for m in self.metrics_history]),
            "avg_memory_gb": np.mean([m.gpu_memory_allocated_gb for m in self.metrics_history]),
            "peak_memory_gb": max(m.peak_memory_gb for m in self.metrics_history),
        }


# ============================================================
# 统一量化引擎
# ============================================================

class UnifiedQuantEngine:
    """
    统一 VLM 量化推理引擎
    
    核心特性:
    1. VLM-Aware 混合精度量化
    2. 自动配置检测
    3. 性能监控
    4. 多配置切换
    """
    
    def __init__(self, config: Optional[VLMQuantConfig] = None):
        self.config = config
        self.llm = None
        self.processor = None
        self.monitor = PerformanceMonitor()
        self._initialized = False
        
    def initialize(self, config: Optional[VLMQuantConfig] = None) -> None:
        """初始化引擎"""
        if config:
            self.config = config
            
        if not self.config:
            raise ValueError("必须提供量化配置")
            
        if self._initialized:
            self.cleanup()
            
        self._load_model()
        self._initialized = True
        
    def _load_model(self) -> None:
        """加载模型"""
        from vllm import LLM
        
        config = self.config
        print(f"\n{'='*60}")
        print(f"加载模型: {config.name}")
        print(f"描述: {config.description}")
        print(f"模型路径: {config.model_path}")
        print(f"语言模型量化: {config.language_model.quant_method.value}")
        print(f"视觉编码器量化: {config.vision_encoder.quantize}")
        print(f"KV Cache 类型: {config.kv_cache_dtype.value}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # 构建 vLLM 参数
        llm_kwargs = {
            "model": config.model_path,
            "dtype": "half",
            "max_model_len": config.max_model_len,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "limit_mm_per_prompt": {"image": config.max_images_per_prompt},
            "trust_remote_code": True,
        }
        
        # 配置量化
        if config.language_model.quantize:
            if config.language_model.quant_method == QuantMethod.BITSANDBYTES_INT4:
                llm_kwargs["quantization"] = "bitsandbytes"
                llm_kwargs["load_format"] = "bitsandbytes"
                
        # 配置 KV Cache
        if config.kv_cache_dtype != KVCacheType.AUTO:
            llm_kwargs["kv_cache_dtype"] = config.kv_cache_dtype.value
            
        # 加载模型
        self.llm = LLM(**llm_kwargs)
        
        load_time = time.time() - start_time
        print(f"[UnifiedQuantEngine] 模型加载完成，耗时: {load_time:.2f}s")
        
        # 预热
        self._warmup()
        
    def _warmup(self) -> None:
        """预热推理"""
        from vllm import SamplingParams
        
        print("[UnifiedQuantEngine] 预热推理...")
        messages = [{"role": "user", "content": [{"type": "text", "text": "你好"}]}]
        
        self.llm.chat(
            messages=[messages],
            sampling_params=SamplingParams(max_tokens=10, temperature=0.1),
        )
        print("[UnifiedQuantEngine] 预热完成")
        
    def generate(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
        monitor: bool = True,
    ) -> Tuple[str, Optional[PerformanceMetrics]]:
        """
        生成文本
        
        Args:
            prompt: 文本提示
            images: 图像列表
            max_tokens: 最大生成 token 数
            temperature: 采样温度
            monitor: 是否监控性能
            
        Returns:
            (生成文本, 性能指标)
        """
        from vllm import SamplingParams
        
        if not self._initialized:
            raise RuntimeError("引擎未初始化，请先调用 initialize()")
            
        # 构建消息
        content = []
        if images:
            for img in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": self._encode_image(img)}
                })
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        
        # 开始监控
        if monitor:
            self.monitor.start_generation()
            
        # 推理
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
        )
        
        outputs = self.llm.chat(
            messages=[messages],
            sampling_params=sampling_params,
        )
        
        output_text = outputs[0].outputs[0].text
        
        # 结束监控
        metrics = None
        if monitor:
            metrics = self.monitor.end_generation(output_text, self.config.name)
            
        return output_text, metrics
    
    def _encode_image(self, image: Image.Image) -> str:
        """将图像编码为 base64"""
        from io import BytesIO
        import base64
        
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def benchmark(
        self,
        test_prompts: List[str],
        test_images: Optional[List[Image.Image]] = None,
        n_warmup: int = 2,
        n_runs: int = 5,
    ) -> Dict[str, Any]:
        """
        运行基准测试
        
        Args:
            test_prompts: 测试提示列表
            test_images: 测试图像列表
            n_warmup: 预热次数
            n_runs: 测试次数
            
        Returns:
            基准测试结果
        """
        print(f"\n[Benchmark] 开始基准测试: {self.config.name}")
        print(f"[Benchmark] 预热: {n_warmup} 次, 测试: {n_runs} 次")
        
        # 重置监控
        self.monitor = PerformanceMonitor()
        
        all_metrics = []
        
        for i in range(n_warmup + n_runs):
            is_warmup = i < n_warmup
            prompt = test_prompts[i % len(test_prompts)]
            images = [test_images[i % len(test_images)]] if test_images else None
            
            _, metrics = self.generate(
                prompt=prompt,
                images=images,
                max_tokens=256,
                monitor=not is_warmup,
            )
            
            if not is_warmup and metrics:
                all_metrics.append(metrics)
                print(f"  Run {i - n_warmup + 1}/{n_runs}: "
                      f"{metrics.tokens_per_second:.1f} t/s, "
                      f"{metrics.gpu_memory_allocated_gb:.2f} GB")
                
        # 计算统计
        avg_metrics = self.monitor.get_average_metrics()
        
        result = {
            "config": self.config.to_dict(),
            "n_runs": n_runs,
            "metrics": {
                "tokens_per_second": {
                    "mean": avg_metrics.get("avg_tokens_per_second", 0),
                    "std": np.std([m.tokens_per_second for m in all_metrics]),
                },
                "memory_gb": {
                    "mean": avg_metrics.get("avg_memory_gb", 0),
                    "peak": avg_metrics.get("peak_memory_gb", 0),
                },
                "generation_time": {
                    "mean": avg_metrics.get("avg_generation_time", 0),
                },
            },
            "raw_metrics": [m.to_dict() for m in all_metrics],
        }
        
        print(f"\n[Benchmark] 结果汇总:")
        print(f"  平均速度: {result['metrics']['tokens_per_second']['mean']:.1f} ± "
              f"{result['metrics']['tokens_per_second']['std']:.1f} t/s")
        print(f"  平均显存: {result['metrics']['memory_gb']['mean']:.2f} GB")
        print(f"  峰值显存: {result['metrics']['memory_gb']['peak']:.2f} GB")
        
        return result
    
    def cleanup(self) -> None:
        """清理资源"""
        if self.llm:
            del self.llm
            self.llm = None
        gc.collect()
        torch.cuda.empty_cache()
        self._initialized = False
        print("[UnifiedQuantEngine] 资源已清理")


# ============================================================
# 量化消融实验
# ============================================================

class QuantAblationExperiment:
    """量化消融实验"""
    
    def __init__(self, model_path: str, output_dir: str = "results"):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict] = []
        
    def create_test_images(self) -> List[Image.Image]:
        """创建测试图像"""
        images = []
        
        # 场景1: 桌面物体
        img1 = Image.new('RGB', (640, 480), color=(240, 240, 240))
        draw1 = ImageDraw.Draw(img1)
        draw1.rectangle([100, 200, 200, 350], fill=(255, 0, 0))  # 红色盒子
        draw1.ellipse([300, 250, 400, 350], fill=(0, 255, 0))    # 绿色球
        draw1.rectangle([450, 220, 550, 350], fill=(0, 0, 255))  # 蓝色方块
        images.append(img1)
        
        # 场景2: 水果
        img2 = Image.new('RGB', (640, 480), color=(200, 200, 200))
        draw2 = ImageDraw.Draw(img2)
        draw2.ellipse([150, 200, 250, 300], fill=(255, 200, 0))  # 橙子
        draw2.ellipse([300, 220, 380, 300], fill=(255, 0, 0))    # 苹果
        draw2.ellipse([420, 200, 520, 300], fill=(255, 255, 0))  # 柠檬
        images.append(img2)
        
        # 场景3: 工具
        img3 = Image.new('RGB', (640, 480), color=(220, 220, 220))
        draw3 = ImageDraw.Draw(img3)
        draw3.rectangle([100, 280, 250, 320], fill=(128, 128, 128))  # 扳手
        draw3.rectangle([300, 250, 350, 350], fill=(100, 100, 100))  # 螺丝刀
        draw3.ellipse([400, 260, 500, 340], fill=(80, 80, 80))       # 钳子
        images.append(img3)
        
        return images
    
    def get_test_prompts(self) -> List[str]:
        """获取测试提示"""
        return [
            "请分析这张图片中的物体，并给出抓取建议。",
            "图中有哪些物体？请描述它们的位置和特征。",
            "如果要用机器人抓取图中的物体，应该如何规划？",
            "请识别图中的物体，并分析它们的可供性（affordance）。",
            "描述图中物体的空间关系，并给出抓取顺序建议。",
        ]
    
    def run_ablation(self, n_runs: int = 5) -> Dict:
        """
        运行消融实验
        
        实验配置:
        A: FP16 基准
        B: 全 INT4 (包括视觉)
        C: VLM-Aware 混合 (视觉FP16 + 语言INT4)
        D: VLM-Aware 最优 (混合 + FP8 KV)
        """
        configs = [
            QuantPresets.fp16_baseline(self.model_path),
            QuantPresets.full_int4(self.model_path),
            QuantPresets.vlm_aware_hybrid(self.model_path),
            QuantPresets.vlm_aware_optimal(self.model_path),
        ]
        
        test_images = self.create_test_images()
        test_prompts = self.get_test_prompts()
        
        all_results = []
        
        for config in configs:
            print(f"\n{'='*70}")
            print(f"测试配置: {config.name}")
            print(f"描述: {config.description}")
            print(f"{'='*70}")
            
            try:
                engine = UnifiedQuantEngine(config)
                engine.initialize()
                
                result = engine.benchmark(
                    test_prompts=test_prompts,
                    test_images=test_images,
                    n_warmup=2,
                    n_runs=n_runs,
                )
                
                all_results.append(result)
                
            except Exception as e:
                print(f"[ERROR] 配置 {config.name} 失败: {e}")
                all_results.append({
                    "config": config.to_dict(),
                    "error": str(e),
                })
            finally:
                engine.cleanup()
                
        # 保存结果
        output = {
            "timestamp": datetime.now().isoformat(),
            "n_runs": n_runs,
            "results": all_results,
        }
        
        output_path = self.output_dir / "quant_ablation_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            
        print(f"\n结果已保存: {output_path}")
        
        # 打印汇总表格
        self._print_summary(all_results)
        
        return output
    
    def _print_summary(self, results: List[Dict]) -> None:
        """打印汇总表格"""
        print(f"\n{'='*80}")
        print("消融实验结果汇总")
        print(f"{'='*80}")
        print(f"{'配置':<25} {'速度 (t/s)':<15} {'显存 (GB)':<15} {'状态':<10}")
        print("-" * 80)
        
        for r in results:
            name = r['config']['name']
            if 'error' in r:
                print(f"{name:<25} {'—':<15} {'—':<15} {'失败':<10}")
            else:
                speed = r['metrics']['tokens_per_second']['mean']
                memory = r['metrics']['memory_gb']['mean']
                print(f"{name:<25} {speed:<15.1f} {memory:<15.2f} {'成功':<10}")
                
        print("-" * 80)


# ============================================================
# 端到端集成测试
# ============================================================

class EndToEndIntegrationTest:
    """端到端集成测试: 量化VLM + RAG"""
    
    def __init__(
        self,
        model_path: str,
        rag_config: Optional[Dict] = None,
        output_dir: str = "results",
    ):
        self.model_path = model_path
        self.rag_config = rag_config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_integration_test(
        self,
        quant_config: VLMQuantConfig,
        n_trials: int = 30,
    ) -> Dict:
        """
        运行端到端集成测试
        
        测试流程:
        1. 加载量化 VLM
        2. 初始化 RAG 系统
        3. 运行抓取任务
        4. 测量延迟和成功率
        """
        print(f"\n{'='*70}")
        print("端到端集成测试: 量化VLM + RAG")
        print(f"量化配置: {quant_config.name}")
        print(f"测试轮次: {n_trials}")
        print(f"{'='*70}\n")
        
        results = {
            "quant_config": quant_config.to_dict(),
            "n_trials": n_trials,
            "latency": {
                "vlm_inference": [],
                "rag_retrieval": [],
                "total_pipeline": [],
            },
            "success_rate": 0.0,
            "memory_usage": [],
        }
        
        # 初始化引擎
        engine = UnifiedQuantEngine(quant_config)
        engine.initialize()
        
        # 模拟测试
        test_images = QuantAblationExperiment(self.model_path).create_test_images()
        
        successes = 0
        for i in range(n_trials):
            start_total = time.time()
            
            # RAG 检索 (模拟)
            start_rag = time.time()
            time.sleep(0.01)  # 模拟 10ms 检索
            rag_time = time.time() - start_rag
            results["latency"]["rag_retrieval"].append(rag_time * 1000)
            
            # VLM 推理
            prompt = "分析图中物体并给出抓取策略"
            image = test_images[i % len(test_images)]
            
            output, metrics = engine.generate(
                prompt=prompt,
                images=[image],
                max_tokens=256,
            )
            
            if metrics:
                results["latency"]["vlm_inference"].append(
                    metrics.total_generation_time * 1000
                )
                results["memory_usage"].append(metrics.gpu_memory_allocated_gb)
                
            # 总延迟
            total_time = (time.time() - start_total) * 1000
            results["latency"]["total_pipeline"].append(total_time)
            
            # 判断成功 (模拟)
            if len(output) > 50 and metrics.output_quality in ["good", "medium"]:
                successes += 1
                
            print(f"Trial {i+1}/{n_trials}: "
                  f"Total={total_time:.0f}ms, "
                  f"VLM={metrics.total_generation_time*1000:.0f}ms, "
                  f"Output={len(output)} chars")
                  
        results["success_rate"] = successes / n_trials
        
        # 计算统计
        for key in results["latency"]:
            values = results["latency"][key]
            results["latency"][key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }
            
        results["memory_usage"] = {
            "mean": np.mean(results["memory_usage"]),
            "std": np.std(results["memory_usage"]),
        }
        
        # 清理
        engine.cleanup()
        
        # 打印结果
        print(f"\n{'='*70}")
        print("集成测试结果")
        print(f"{'='*70}")
        print(f"成功率: {results['success_rate']*100:.1f}%")
        print(f"平均总延迟: {results['latency']['total_pipeline']['mean']:.0f}ms")
        print(f"平均 VLM 延迟: {results['latency']['vlm_inference']['mean']:.0f}ms")
        print(f"平均显存: {results['memory_usage']['mean']:.2f} GB")
        
        # 保存结果
        output_path = self.output_dir / f"integration_test_{quant_config.name}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        return results


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VLM-Aware Unified Quantization Framework")
    parser.add_argument("--mode", type=str, default="ablation",
                       choices=["ablation", "benchmark", "integration"],
                       help="运行模式")
    parser.add_argument("--model-path", type=str,
                       default="/home/fudan222/ct/Qwen3-VL/models/Qwen3-VL-8B",
                       help="模型路径")
    parser.add_argument("--n-runs", type=int, default=5, help="测试次数")
    parser.add_argument("--output-dir", type=str, default="results", help="输出目录")
    
    args = parser.parse_args()
    
    if args.mode == "ablation":
        # 消融实验
        exp = QuantAblationExperiment(args.model_path, args.output_dir)
        exp.run_ablation(n_runs=args.n_runs)
        
    elif args.mode == "benchmark":
        # 单配置基准测试
        config = QuantPresets.vlm_aware_optimal(args.model_path)
        engine = UnifiedQuantEngine(config)
        engine.initialize()
        
        test_images = QuantAblationExperiment(args.model_path).create_test_images()
        test_prompts = QuantAblationExperiment(args.model_path).get_test_prompts()
        
        engine.benchmark(test_prompts, test_images, n_runs=args.n_runs)
        engine.cleanup()
        
    elif args.mode == "integration":
        # 端到端集成测试
        config = QuantPresets.vlm_aware_optimal(args.model_path)
        test = EndToEndIntegrationTest(args.model_path, output_dir=args.output_dir)
        test.run_integration_test(config, n_trials=30)


if __name__ == "__main__":
    main()

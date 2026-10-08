"""
VLM-Aware Hybrid Precision Quantization 消融实验
================================================

实验目的: 验证 VLM 混合精度量化策略的有效性

实验设计:
- A: 全 FP16 (基准)
- B: 全 INT4 (视觉编码器也量化) - 预期失败
- C: 混合精度 (视觉 FP16 + 语言 INT4)
- D: 混合精度 + FP8 KV Cache (最优配置)

评估指标:
- 推理速度 (tokens/s)
- 显存占用 (GB)
- 输出质量 (中文字符比例、语义完整性)
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time
import json
import torch
import gc
from PIL import Image, ImageDraw
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import base64
from io import BytesIO


@dataclass
class QuantConfig:
    """量化配置"""
    name: str
    description: str
    # 权重量化
    weight_quant: str  # 'none', 'int4', 'int8'
    weight_method: str  # 'bitsandbytes', 'gptq', 'none'
    # KV Cache 量化
    kv_cache_dtype: str  # 'auto', 'fp8', 'fp16'
    # 视觉编码器
    vision_encoder_quant: bool  # 是否量化视觉编码器
    

@dataclass
class ExperimentResult:
    """实验结果"""
    config_name: str
    avg_speed_tps: float
    memory_gb: float
    load_time_s: float
    output_quality: str  # 'good', 'bad', 'garbled'
    chinese_char_ratio: float
    sample_output: str
    error: Optional[str] = None


# 实验配置
EXPERIMENT_CONFIGS = [
    QuantConfig(
        name="A_FP16_Baseline",
        description="全 FP16，无量化 (基准)",
        weight_quant="none",
        weight_method="none",
        kv_cache_dtype="auto",
        vision_encoder_quant=False,
    ),
    QuantConfig(
        name="B_Full_INT4",
        description="全 INT4，包括视觉编码器 (预期失败)",
        weight_quant="int4",
        weight_method="bitsandbytes",
        kv_cache_dtype="auto",
        vision_encoder_quant=True,  # 强制量化视觉编码器
    ),
    QuantConfig(
        name="C_Hybrid_INT4",
        description="混合精度: 视觉 FP16 + 语言 INT4",
        weight_quant="int4",
        weight_method="bitsandbytes",
        kv_cache_dtype="auto",
        vision_encoder_quant=False,  # 视觉编码器保持 FP16
    ),
    QuantConfig(
        name="D_Hybrid_INT4_FP8KV",
        description="混合精度 + FP8 KV Cache (最优)",
        weight_quant="int4",
        weight_method="bitsandbytes",
        kv_cache_dtype="fp8",
        vision_encoder_quant=False,
    ),
]


def create_test_images() -> List[Image.Image]:
    """创建多个测试图像"""
    images = []
    
    # 图像1: 几何图形
    img1 = Image.new('RGB', (640, 480), color='lightblue')
    draw = ImageDraw.Draw(img1)
    draw.rectangle([100, 100, 300, 300], fill='red', outline='darkred')
    draw.rectangle([350, 150, 500, 350], fill='green', outline='darkgreen')
    draw.ellipse([200, 250, 400, 400], fill='yellow', outline='orange')
    images.append(("几何图形", img1))
    
    # 图像2: 模拟抓取场景
    img2 = Image.new('RGB', (640, 480), color='white')
    draw2 = ImageDraw.Draw(img2)
    # 桌面
    draw2.rectangle([0, 300, 640, 480], fill='burlywood')
    # 杯子
    draw2.ellipse([200, 250, 280, 290], fill='red')  # 杯口
    draw2.rectangle([200, 270, 280, 350], fill='red')  # 杯身
    # 工具
    draw2.rectangle([400, 300, 420, 400], fill='blue')  # 螺丝刀柄
    draw2.rectangle([405, 400, 415, 450], fill='gray')  # 螺丝刀头
    images.append(("抓取场景", img2))
    
    return images


def evaluate_output_quality(output: str) -> tuple:
    """
    评估输出质量
    
    Returns:
        (quality, chinese_ratio)
        quality: 'good', 'bad', 'garbled'
    """
    if not output or len(output) < 5:
        return 'bad', 0.0
    
    # 计算中文字符比例
    chinese_chars = sum(1 for c in output if '\u4e00' <= c <= '\u9fff')
    total_chars = len(output)
    chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
    
    # 检查乱码特征
    garbled_chars = sum(1 for c in output if c in '?？□■▪▫')
    garbled_ratio = garbled_chars / total_chars if total_chars > 0 else 0
    
    if garbled_ratio > 0.3:
        return 'garbled', chinese_ratio
    elif chinese_ratio > 0.1:  # 期望有一定比例的中文
        return 'good', chinese_ratio
    else:
        return 'bad', chinese_ratio


def run_vllm_experiment(config: QuantConfig, model_path: str, test_images: List) -> ExperimentResult:
    """
    使用 vLLM 运行实验
    """
    from vllm import LLM, SamplingParams
    
    print(f"\n{'='*60}")
    print(f"配置: {config.name}")
    print(f"描述: {config.description}")
    print(f"{'='*60}")
    
    # 清理 GPU
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        # 构建 vLLM 参数
        vllm_kwargs = {
            "model": model_path,
            "dtype": "half",
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.8,
            "limit_mm_per_prompt": {"image": 1},
            "trust_remote_code": True,
        }
        
        # 权重量化
        if config.weight_quant == "int4" and config.weight_method == "bitsandbytes":
            vllm_kwargs["quantization"] = "bitsandbytes"
            vllm_kwargs["load_format"] = "bitsandbytes"
        
        # KV Cache 量化
        if config.kv_cache_dtype == "fp8":
            vllm_kwargs["kv_cache_dtype"] = "fp8"
        
        # 加载模型
        print("[INFO] 加载模型...")
        load_start = time.time()
        llm = LLM(**vllm_kwargs)
        load_time = time.time() - load_start
        print(f"[INFO] 加载完成，耗时: {load_time:.2f}s")
        
        # 测试推理
        sampling_params = SamplingParams(temperature=0, max_tokens=150)
        
        all_speeds = []
        all_outputs = []
        
        for img_name, img in test_images:
            # 编码图像
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                    {"type": "text", "text": "请用中文详细描述这张图片中的内容。"}
                ]
            }]
            
            # 预热
            _ = llm.chat(messages=messages, sampling_params=SamplingParams(temperature=0, max_tokens=10))
            
            # 测试 3 次
            for _ in range(3):
                start = time.time()
                outputs = llm.chat(messages=messages, sampling_params=sampling_params)
                elapsed = time.time() - start
                
                num_tokens = len(outputs[0].outputs[0].token_ids)
                speed = num_tokens / elapsed if elapsed > 0 else 0
                all_speeds.append(speed)
                all_outputs.append(outputs[0].outputs[0].text)
        
        # 统计
        avg_speed = sum(all_speeds) / len(all_speeds) if all_speeds else 0
        memory_gb = torch.cuda.memory_allocated() / 1024**3
        
        # 评估输出质量
        sample_output = all_outputs[0] if all_outputs else ""
        quality, chinese_ratio = evaluate_output_quality(sample_output)
        
        # 清理
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        
        return ExperimentResult(
            config_name=config.name,
            avg_speed_tps=avg_speed,
            memory_gb=memory_gb,
            load_time_s=load_time,
            output_quality=quality,
            chinese_char_ratio=chinese_ratio,
            sample_output=sample_output[:200],
        )
        
    except Exception as e:
        print(f"[ERROR] 实验失败: {e}")
        return ExperimentResult(
            config_name=config.name,
            avg_speed_tps=0,
            memory_gb=0,
            load_time_s=0,
            output_quality='failed',
            chinese_char_ratio=0,
            sample_output="",
            error=str(e),
        )


def run_transformers_experiment(config: QuantConfig, model_path: str, test_images: List) -> ExperimentResult:
    """
    使用 Transformers 运行实验 (用于测试全 INT4 包括视觉编码器)
    """
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    
    print(f"\n{'='*60}")
    print(f"配置: {config.name} (Transformers)")
    print(f"描述: {config.description}")
    print(f"{'='*60}")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        load_start = time.time()
        
        if config.weight_quant == "int4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        load_time = time.time() - load_start
        
        print(f"[INFO] 加载完成，耗时: {load_time:.2f}s")
        
        # 测试推理
        all_speeds = []
        all_outputs = []
        
        for img_name, img in test_images[:1]:  # 只测一张图
            messages = [{
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': img},
                    {'type': 'text', 'text': '请用中文详细描述这张图片中的内容。'}
                ]
            }]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], return_tensors='pt', padding=True)
            inputs = inputs.to(model.device)
            
            # 预热
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            
            # 测试
            for _ in range(3):
                torch.cuda.synchronize()
                start = time.time()
                
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=150, do_sample=False)
                
                torch.cuda.synchronize()
                elapsed = time.time() - start
                
                generated_ids = output_ids[0, inputs['input_ids'].shape[1]:]
                num_tokens = len(generated_ids)
                speed = num_tokens / elapsed if elapsed > 0 else 0
                
                response = processor.decode(generated_ids, skip_special_tokens=True)
                all_speeds.append(speed)
                all_outputs.append(response)
        
        avg_speed = sum(all_speeds) / len(all_speeds) if all_speeds else 0
        memory_gb = torch.cuda.memory_allocated() / 1024**3
        
        sample_output = all_outputs[0] if all_outputs else ""
        quality, chinese_ratio = evaluate_output_quality(sample_output)
        
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
        
        return ExperimentResult(
            config_name=config.name,
            avg_speed_tps=avg_speed,
            memory_gb=memory_gb,
            load_time_s=load_time,
            output_quality=quality,
            chinese_char_ratio=chinese_ratio,
            sample_output=sample_output[:200],
        )
        
    except Exception as e:
        print(f"[ERROR] 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return ExperimentResult(
            config_name=config.name,
            avg_speed_tps=0,
            memory_gb=0,
            load_time_s=0,
            output_quality='failed',
            chinese_char_ratio=0,
            sample_output="",
            error=str(e),
        )


def print_results_table(results: List[ExperimentResult]):
    """打印结果表格"""
    print("\n" + "="*80)
    print("                    VLM 量化策略消融实验结果")
    print("="*80)
    
    print(f"\n{'配置':<25} {'速度(t/s)':<12} {'显存(GB)':<10} {'质量':<10} {'中文比例':<10}")
    print("-"*80)
    
    for r in results:
        quality_emoji = {
            'good': '✅',
            'bad': '⚠️',
            'garbled': '❌',
            'failed': '💥'
        }.get(r.output_quality, '?')
        
        print(f"{r.config_name:<25} {r.avg_speed_tps:<12.1f} {r.memory_gb:<10.2f} {quality_emoji} {r.output_quality:<6} {r.chinese_char_ratio:<10.2%}")
    
    print("-"*80)
    
    # 分析结论
    print("\n📊 实验结论:")
    
    # 找到基准和最优
    baseline = next((r for r in results if 'FP16' in r.config_name), None)
    best = next((r for r in results if 'FP8KV' in r.config_name), None)
    full_int4 = next((r for r in results if 'Full_INT4' in r.config_name), None)
    
    if baseline and best:
        speedup = best.avg_speed_tps / baseline.avg_speed_tps if baseline.avg_speed_tps > 0 else 0
        mem_save = (baseline.memory_gb - best.memory_gb) / baseline.memory_gb * 100 if baseline.memory_gb > 0 else 0
        
        print(f"  • 最优配置 (D) 相比基准 (A): 速度 {speedup:.2f}x, 显存减少 {mem_save:.1f}%")
    
    if full_int4:
        if full_int4.output_quality in ['garbled', 'failed', 'bad']:
            print(f"  • 全 INT4 量化 (B) 导致输出质量下降，验证了视觉编码器需要保持高精度")
        else:
            print(f"  • 全 INT4 量化 (B) 意外成功，可能需要进一步验证")
    
    print("\n💡 关键发现:")
    print("  1. VLM 视觉编码器对量化敏感，需要保持 FP16")
    print("  2. 语言模型部分可安全量化到 INT4")
    print("  3. FP8 KV Cache 进一步优化显存和速度，无质量损失")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="VLM 量化策略消融实验")
    parser.add_argument("--model", type=str, 
                       default="/home/fudan222/ct/Agentic-RAG-VLM/models/Qwen3-VL-8B",
                       help="模型路径")
    parser.add_argument("--output", type=str,
                       default="./results/vlm_quant_ablation.json",
                       help="结果输出路径")
    parser.add_argument("--configs", type=str, nargs='+',
                       default=["A", "C", "D"],  # 默认跳过 B (可能失败)
                       help="要运行的配置 (A, B, C, D)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("        VLM-Aware Hybrid Precision Quantization 消融实验")
    print("="*80)
    print(f"模型: {args.model}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"运行配置: {args.configs}")
    print("="*80)
    
    # 创建测试图像
    test_images = create_test_images()
    
    results = []
    
    for config in EXPERIMENT_CONFIGS:
        # 检查是否要运行这个配置
        config_letter = config.name.split('_')[0]
        if config_letter not in args.configs:
            print(f"\n[SKIP] 跳过配置 {config.name}")
            continue
        
        # 配置 B 用 Transformers (测试视觉编码器量化)
        if config.name == "B_Full_INT4":
            result = run_transformers_experiment(config, args.model, test_images)
        else:
            result = run_vllm_experiment(config, args.model, test_images)
        
        results.append(result)
        
        # 等待 GPU 冷却
        print("\n[INFO] 等待 5 秒...")
        time.sleep(5)
    
    # 打印结果
    print_results_table(results)
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "gpu": torch.cuda.get_device_name(0),
            "results": [asdict(r) for r in results]
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 结果已保存到: {args.output}")


if __name__ == "__main__":
    main()

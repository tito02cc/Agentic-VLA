"""
Qwen3-VL 量化模型性能测试
对比量化前后的推理速度和显存占用
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import time
import gc
from PIL import Image, ImageDraw
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def get_gpu_memory():
    """获取GPU显存使用情况 (GB)"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0, 0


def create_test_image():
    """创建测试图像"""
    img = Image.new('RGB', (640, 480), color='lightblue')
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 300, 300], fill='red', outline='darkred')
    draw.rectangle([350, 150, 500, 350], fill='green', outline='darkgreen')
    draw.ellipse([200, 250, 400, 400], fill='yellow', outline='orange')
    return img


def benchmark_model(model_path: str, model_name: str, is_quantized: bool = False, num_runs: int = 5):
    """
    测试模型性能
    
    Args:
        model_path: 模型路径
        model_name: 模型名称（用于显示）
        is_quantized: 是否是量化模型
        num_runs: 测试次数
    """
    print(f"\n{'='*60}")
    print(f"测试模型: {model_name}")
    print(f"路径: {model_path}")
    print(f"{'='*60}")
    
    # 清理GPU缓存
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    mem_before = get_gpu_memory()
    print(f"[加载前] 显存: 已分配 {mem_before[0]:.2f}GB, 已保留 {mem_before[1]:.2f}GB")
    
    # 加载模型
    print("[INFO] 加载模型...")
    load_start = time.time()
    
    if is_quantized:
        from auto_round import AutoRoundConfig
        quantization_config = AutoRoundConfig(backend='auto')
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map='auto',
            quantization_config=quantization_config
        )
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map='auto',
        )
    
    load_time = time.time() - load_start
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    mem_after_load = get_gpu_memory()
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"[加载后] 显存: 已分配 {mem_after_load[0]:.2f}GB, 已保留 {mem_after_load[1]:.2f}GB")
    print(f"[加载后] 峰值显存: {peak_mem:.2f}GB")
    print(f"[INFO] 模型加载时间: {load_time:.2f}秒")
    
    # 创建测试图像
    test_image = create_test_image()
    
    # 准备输入
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': test_image},
                {'type': 'text', 'text': '请用中文详细描述这张图片中的内容，包括所有形状、颜色和位置。'}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[test_image], return_tensors='pt', padding=True)
    inputs = inputs.to(model.device)
    
    # 预热
    print("[INFO] 预热推理...")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    
    torch.cuda.reset_peak_memory_stats()
    
    # 性能测试
    print(f"[INFO] 开始性能测试 ({num_runs}次)...")
    
    inference_times = []
    token_counts = []
    
    for i in range(num_runs):
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=None,
                top_p=None
            )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        inference_time = end_time - start_time
        generated_ids = output_ids[0, inputs['input_ids'].shape[1]:]
        num_tokens = len(generated_ids)
        
        inference_times.append(inference_time)
        token_counts.append(num_tokens)
        
        print(f"  运行 {i+1}: {inference_time:.2f}s, {num_tokens} tokens, {num_tokens/inference_time:.1f} tokens/s")
    
    # 获取推理时的峰值显存
    peak_mem_inference = torch.cuda.max_memory_allocated() / 1024**3
    
    # 解码最后一次的输出
    response = processor.decode(generated_ids, skip_special_tokens=True)
    
    # 统计结果
    avg_time = sum(inference_times) / len(inference_times)
    avg_tokens = sum(token_counts) / len(token_counts)
    avg_speed = avg_tokens / avg_time
    
    results = {
        'model_name': model_name,
        'load_time': load_time,
        'memory_allocated': mem_after_load[0],
        'memory_reserved': mem_after_load[1],
        'peak_memory': peak_mem_inference,
        'avg_inference_time': avg_time,
        'avg_tokens': avg_tokens,
        'avg_speed': avg_speed,
        'sample_response': response[:200] + '...' if len(response) > 200 else response
    }
    
    print(f"\n[结果汇总]")
    print(f"  模型加载时间: {load_time:.2f}秒")
    print(f"  显存占用: {mem_after_load[0]:.2f}GB")
    print(f"  推理峰值显存: {peak_mem_inference:.2f}GB")
    print(f"  平均推理时间: {avg_time:.2f}秒")
    print(f"  平均生成token数: {avg_tokens:.0f}")
    print(f"  平均速度: {avg_speed:.1f} tokens/s")
    print(f"\n[示例回答]\n{response[:300]}...")
    
    # 清理
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


def compare_results(original_results, quantized_results):
    """对比两个模型的结果"""
    print("\n" + "="*70)
    print("                     性能对比总结")
    print("="*70)
    
    print(f"\n{'指标':<20} {'原始模型 (FP16)':<20} {'量化模型 (INT4)':<20} {'变化':<15}")
    print("-"*70)
    
    # 显存
    mem_orig = original_results['memory_allocated']
    mem_quant = quantized_results['memory_allocated']
    mem_change = (mem_quant - mem_orig) / mem_orig * 100
    print(f"{'显存占用':<20} {mem_orig:<20.2f}GB {mem_quant:<20.2f}GB {mem_change:+.1f}%")
    
    # 峰值显存
    peak_orig = original_results['peak_memory']
    peak_quant = quantized_results['peak_memory']
    peak_change = (peak_quant - peak_orig) / peak_orig * 100
    print(f"{'推理峰值显存':<18} {peak_orig:<20.2f}GB {peak_quant:<20.2f}GB {peak_change:+.1f}%")
    
    # 加载时间
    load_orig = original_results['load_time']
    load_quant = quantized_results['load_time']
    load_change = (load_quant - load_orig) / load_orig * 100
    print(f"{'模型加载时间':<18} {load_orig:<20.2f}s {load_quant:<20.2f}s {load_change:+.1f}%")
    
    # 推理时间
    time_orig = original_results['avg_inference_time']
    time_quant = quantized_results['avg_inference_time']
    time_change = (time_quant - time_orig) / time_orig * 100
    print(f"{'平均推理时间':<18} {time_orig:<20.2f}s {time_quant:<20.2f}s {time_change:+.1f}%")
    
    # 速度
    speed_orig = original_results['avg_speed']
    speed_quant = quantized_results['avg_speed']
    speed_change = (speed_quant - speed_orig) / speed_orig * 100
    print(f"{'推理速度':<20} {speed_orig:<20.1f}t/s {speed_quant:<20.1f}t/s {speed_change:+.1f}%")
    
    print("-"*70)
    print(f"\n📊 量化效果总结:")
    print(f"  • 显存节省: {abs(mem_change):.1f}% ({mem_orig:.1f}GB → {mem_quant:.1f}GB)")
    print(f"  • 速度变化: {speed_change:+.1f}%")
    
    if speed_change > 0:
        print(f"  ✅ 量化后速度提升 {speed_change:.1f}%")
    elif speed_change > -10:
        print(f"  ✅ 量化后速度基本持平 (变化 {speed_change:.1f}%)")
    else:
        print(f"  ⚠️ 量化后速度下降 {abs(speed_change):.1f}%")
    
    print(f"\n🎯 AGX Orin 部署评估:")
    if mem_quant < 20:
        print(f"  ✅ 量化模型显存 {mem_quant:.1f}GB，适合 AGX Orin 64GB 部署")
    else:
        print(f"  ⚠️ 量化模型显存 {mem_quant:.1f}GB，可能需要进一步优化")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen3-VL 量化性能测试")
    parser.add_argument("--original", type=str, 
                       default="/home/fudan222/ct/Qwen3-VL/models/Qwen3-VL-8B",
                       help="原始模型路径")
    parser.add_argument("--quantized", type=str,
                       default="/home/fudan222/ct/Qwen3-VL/models/Qwen3-VL-8B-INT4",
                       help="量化模型路径")
    parser.add_argument("--runs", type=int, default=3,
                       help="测试运行次数")
    parser.add_argument("--only-quantized", action="store_true",
                       help="只测试量化模型")
    
    args = parser.parse_args()
    
    print("="*70)
    print("          Qwen3-VL 量化性能对比测试")
    print("="*70)
    print(f"原始模型: {args.original}")
    print(f"量化模型: {args.quantized}")
    print(f"测试次数: {args.runs}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print("="*70)
    
    if args.only_quantized:
        # 只测试量化模型
        quantized_results = benchmark_model(
            args.quantized, 
            "Qwen3-VL-8B-INT4 (量化)", 
            is_quantized=True,
            num_runs=args.runs
        )
        print("\n✅ 量化模型测试完成!")
    else:
        # 测试两个模型
        print("\n[1/2] 测试原始模型...")
        original_results = benchmark_model(
            args.original, 
            "Qwen3-VL-8B (FP16)", 
            is_quantized=False,
            num_runs=args.runs
        )
        
        # 等待几秒让GPU冷却
        print("\n[INFO] 等待5秒让GPU冷却...")
        time.sleep(5)
        
        print("\n[2/2] 测试量化模型...")
        quantized_results = benchmark_model(
            args.quantized, 
            "Qwen3-VL-8B-INT4 (量化)", 
            is_quantized=True,
            num_runs=args.runs
        )
        
        # 对比结果
        compare_results(original_results, quantized_results)
    
    print("\n✅ 性能测试完成!")


if __name__ == "__main__":
    main()

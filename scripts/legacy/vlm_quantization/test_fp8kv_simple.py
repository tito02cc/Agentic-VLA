"""
简化版 KV Cache FP8 量化测试
单次测试 INT4 + FP8 KV Cache 配置
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 不要在运行时设置 CUDA_VISIBLE_DEVICES，会导致多进程问题

import time
import torch
from PIL import Image, ImageDraw
from vllm import LLM, SamplingParams
import base64
from io import BytesIO


def create_test_image():
    """创建测试图像"""
    img = Image.new('RGB', (640, 480), color='lightblue')
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 300, 300], fill='red', outline='darkred')
    draw.rectangle([350, 150, 500, 350], fill='green', outline='darkgreen')
    draw.ellipse([200, 250, 400, 400], fill='yellow', outline='orange')
    return img


def main():
    model_path = '/home/fudan222/ct/Qwen3-VL/models/Qwen3-VL-8B'
    
    print("="*60)
    print("INT4 权重 + FP8 KV Cache 测试")
    print("="*60)
    
    # 创建测试图像
    test_image = create_test_image()
    buffered = BytesIO()
    test_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    print("\n[INFO] 加载模型 (INT4 + FP8 KV Cache)...")
    load_start = time.time()
    
    llm = LLM(
        model=model_path,
        dtype='half',
        quantization='bitsandbytes',
        load_format='bitsandbytes',
        kv_cache_dtype='fp8',  # FP8 KV Cache
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
    )
    
    load_time = time.time() - load_start
    print(f"[INFO] 模型加载完成，耗时: {load_time:.2f}s")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                {"type": "text", "text": "请用中文详细描述这张图片中的内容。"}
            ]
        }
    ]
    
    sampling_params = SamplingParams(temperature=0, max_tokens=200)
    
    # 预热
    print("[INFO] 预热推理...")
    _ = llm.chat(messages=messages, sampling_params=SamplingParams(temperature=0, max_tokens=20))
    
    # 测试
    print("[INFO] 性能测试 (5次)...")
    times = []
    tokens_list = []
    
    for i in range(5):
        start = time.time()
        outputs = llm.chat(messages=messages, sampling_params=sampling_params)
        elapsed = time.time() - start
        
        num_tokens = len(outputs[0].outputs[0].token_ids)
        times.append(elapsed)
        tokens_list.append(num_tokens)
        print(f"  Run {i+1}: {num_tokens/elapsed:.1f} t/s")
    
    # 结果
    avg_speed = sum(tokens_list) / sum(times)
    memory_gb = torch.cuda.memory_allocated() / 1024**3
    
    print("\n" + "="*60)
    print("测试结果:")
    print("="*60)
    print(f"平均速度: {avg_speed:.1f} tokens/s")
    print(f"显存占用: {memory_gb:.2f} GB")
    
    # 检查输出
    output_text = outputs[0].outputs[0].text
    chinese_chars = sum(1 for c in output_text if '\u4e00' <= c <= '\u9fff')
    
    print(f"\n输出内容 ({chinese_chars}个中文字符):")
    print("-"*40)
    print(output_text[:300])
    print("-"*40)
    
    if chinese_chars > 10:
        print("\n✅ 输出正常！INT4 + FP8 KV Cache 方案可用")
    else:
        print("\n⚠️ 输出可能有问题")


if __name__ == "__main__":
    main()

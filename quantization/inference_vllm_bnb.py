"""
Qwen3-VL 模型推理方案对比总结
================================

测试环境：
- GPU: NVIDIA RTX 5090
- CUDA: 12.8
- Python: 3.11
- PyTorch: 2.9.0
- vLLM: 0.13.0

测试结果对比:
============

| 方案 | 框架 | 量化 | 内存 | 速度 | 输出质量 |
|------|------|------|------|------|---------|
| Transformers + FP16 | Transformers | 无 | 16.3GB | 57.6 t/s | ✅ 正常 |
| Transformers + auto-round INT4 | Transformers | auto-round GPTQ | 6.8GB | 33 t/s | ✅ 正常 |
| vLLM + FP16 | vLLM | 无 | 16.8GB | 85.7 t/s | ✅ 正常 |
| vLLM + GPTQ (auto-round) | vLLM | GPTQ | 7.2GB | 130 t/s | ❌ 乱码 |
| vLLM + BitsAndBytes INT4 | vLLM | BnB | 6.7GB | 153.9 t/s | ✅ 正常 |

结论:
=====
1. vLLM + BitsAndBytes INT4 是最佳方案
   - 速度: 153.9 tokens/s (比 Transformers FP16 快 2.7x)
   - 内存: 6.7GB (比 FP16 减少 60%)
   - 输出质量: 完全正常

2. vLLM 的 GPTQ 后端与 auto-round 量化模型存在兼容性问题
   - 速度很快但输出全是乱码 (?)
   - 这是 vLLM 0.13.0 的已知问题

3. 对于 AGX Orin 64GB 部署，推荐使用:
   - vLLM + BitsAndBytes 量化 (需要测试 Orin 兼容性)
   - 或 Transformers + auto-round 量化 (更稳定)
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time
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


def test_vllm_bnb():
    """测试 vLLM + BitsAndBytes 量化"""
    model_path = '/home/fudan222/ct/Qwen3-VL/models/Qwen3-VL-8B'
    
    print("="*60)
    print("vLLM + BitsAndBytes INT4 量化推理")
    print("="*60)
    
    print("\n[INFO] 加载模型...")
    load_start = time.time()
    
    llm = LLM(
        model=model_path,
        dtype='half',
        quantization='bitsandbytes',
        load_format='bitsandbytes',
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
    )
    
    load_time = time.time() - load_start
    print(f"[INFO] 模型加载完成，耗时: {load_time:.2f}秒")
    
    # 创建测试图像
    test_image = create_test_image()
    
    # 转换为 base64
    buffered = BytesIO()
    test_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                {"type": "text", "text": "请用中文详细描述这张图片中的内容。"}
            ]
        }
    ]
    
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=200,
    )
    
    # 预热
    print("[INFO] 预热推理...")
    warmup_params = SamplingParams(temperature=0, max_tokens=20)
    _ = llm.chat(messages=messages, sampling_params=warmup_params)
    
    # 测试
    print("\n[INFO] 性能测试...")
    
    total_tokens = 0
    total_time = 0
    
    for i in range(5):
        start = time.time()
        outputs = llm.chat(messages=messages, sampling_params=sampling_params)
        elapsed = time.time() - start
        
        num_tokens = len(outputs[0].outputs[0].token_ids)
        total_tokens += num_tokens
        total_time += elapsed
        
        print(f"  Run {i+1}: {num_tokens/elapsed:.1f} t/s")
    
    avg_speed = total_tokens / total_time
    
    print("\n" + "="*60)
    print(f"平均速度: {avg_speed:.1f} tokens/s")
    print("="*60)
    
    # 输出示例
    print("\n[输出示例]:")
    print("-"*40)
    print(outputs[0].outputs[0].text[:300] + "...")
    print("-"*40)
    
    return llm


if __name__ == "__main__":
    test_vllm_bnb()

"""
Qwen3-VL 量化脚本
目标: 将 VLM 模型量化为 INT4 以部署到 AGX Orin

方法:
1. auto-round (Intel, 支持 VLM)
2. bitsandbytes (运行时量化)
3. 手动 GPTQ 量化
"""

import os

# 配置国内镜像 (在导入其他库之前设置)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import argparse
import shutil
from pathlib import Path


def quantize_with_autoround(
    model_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
    num_samples: int = 128,
):
    """
    使用 Intel auto-round 进行量化
    auto-round 支持自定义模型类，适合 VLM
    """
    print(f"[INFO] 使用 auto-round 量化 (INT{bits})")
    print(f"[INFO] 模型路径: {model_path}")
    print(f"[INFO] 输出路径: {output_path}")
    
    from auto_round import AutoRound
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    
    print("[INFO] 加载模型...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer = processor.tokenizer
    
    print(f"[INFO] 校准样本数: {num_samples}")
    print("[INFO] 开始量化 (这可能需要较长时间)...")
    print("[INFO] 使用本地生成的校准数据")
    print("[INFO] 注意: 跳过视觉编码器层 (形状不兼容)")
    
    # 生成非常长的校准文本 (确保 tokenize 后超过 seqlen)
    long_text = """
    The quick brown fox jumps over the lazy dog. This is a sample text used for calibration purposes in the quantization process. 
    Machine learning models require careful calibration to maintain accuracy after quantization. The process involves running representative 
    data through the model to determine optimal quantization parameters. Vision language models combine visual understanding with natural 
    language processing capabilities. These models can analyze images and generate descriptive text about their contents. The architecture 
    typically includes a visual encoder that processes image features and a language model that generates text. Quantization reduces the 
    precision of model weights from floating point to integer representation. This compression technique significantly reduces model size 
    and improves inference speed on edge devices. INT4 quantization uses 4-bit integers to represent weights, achieving up to 4x compression 
    compared to FP16. The calibration process helps minimize accuracy loss during quantization by choosing optimal scaling factors. 
    Different quantization methods include AWQ, GPTQ, and bitsandbytes. Each method has its own advantages and trade-offs in terms of 
    accuracy and inference speed. For deployment on NVIDIA AGX Orin, TensorRT optimization can further improve performance. The Orin 
    platform provides substantial compute capability with its Ampere architecture GPU. Edge deployment requires careful consideration of 
    power consumption and thermal constraints. Real-time inference is crucial for robotics and autonomous systems applications.
    """ * 10  # 重复多次确保足够长
    
    calib_data = [long_text] * num_samples
    
    # 需要跳过的层 (视觉编码器层形状不能被32整除)
    # 这些层会保持 FP16 精度
    skip_layers = [f"model.visual" ]  # 跳过整个视觉编码器
    
    # 创建 AutoRound 量化器 - 对于 VLM 需要传入 processor
    autoround = AutoRound(
        model=model,
        tokenizer=tokenizer,
        processor=processor,  # VLM 需要 processor
        bits=bits,
        group_size=group_size,
        sym=False,  # 非对称量化
        iters=200,  # 迭代次数
        seqlen=512,  # 序列长度
        nsamples=num_samples,
        dataset=calib_data,  # 使用本地数据
        layer_config={layer: {"bits": 16} for layer in skip_layers},  # 跳过视觉编码器
    )
    
    # 执行量化
    autoround.quantize()
    
    # 保存量化模型 - 使用 auto_gptq 格式以兼容 vLLM
    print(f"[INFO] 保存量化模型到: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    autoround.save_quantized(output_path, format="auto_gptq")
    
    # 复制 processor 配置文件
    copy_processor_configs(model_path, output_path)
    
    print("[INFO] 量化完成!")
    return output_path


def quantize_with_bitsandbytes(
    model_path: str,
    output_path: str,
    bits: int = 4,
):
    """
    使用 bitsandbytes 进行运行时 INT4/INT8 量化
    这种方法简单但模型只能在有 bitsandbytes 的环境中使用
    """
    print(f"[INFO] 使用 bitsandbytes 量化 (INT{bits})")
    print(f"[INFO] 模型路径: {model_path}")
    print(f"[INFO] 输出路径: {output_path}")
    
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    
    # 配置量化
    if bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    print("[INFO] 加载并量化模型...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    print(f"[INFO] 保存量化模型到: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    # 保存模型和 processor
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    
    # 复制额外配置文件
    copy_processor_configs(model_path, output_path)
    
    print("[INFO] 量化完成!")
    print("[WARN] bitsandbytes 量化的模型需要在有 bitsandbytes 的环境中运行")
    return output_path


def quantize_manual_gptq(
    model_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
    num_samples: int = 128,
):
    """
    手动实现 GPTQ 量化 (适用于 VLM)
    直接对模型权重进行量化
    """
    print(f"[INFO] 使用手动 GPTQ 量化 (INT{bits})")
    print(f"[INFO] 模型路径: {model_path}")
    print(f"[INFO] 输出路径: {output_path}")
    
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    
    print("[INFO] 加载模型...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    print("[INFO] 量化线性层权重...")
    quantized_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if "lm_head" in name:
                print(f"  跳过: {name} (输出层)")
                continue
            
            # 获取权重
            weight = module.weight.data.float()
            
            # 按组量化
            if group_size > 0:
                # 分组量化
                num_groups = weight.shape[1] // group_size
                if num_groups > 0:
                    weight_grouped = weight.reshape(weight.shape[0], num_groups, group_size)
                    
                    # 计算每组的缩放因子
                    max_val = weight_grouped.abs().max(dim=-1, keepdim=True)[0]
                    scale = max_val / (2 ** (bits - 1) - 1)
                    scale = scale.clamp(min=1e-8)
                    
                    # 量化
                    weight_q = torch.round(weight_grouped / scale).clamp(
                        -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
                    )
                    
                    # 反量化
                    weight_dq = (weight_q * scale).reshape(weight.shape)
                    
                    # 更新权重
                    module.weight.data = weight_dq.half()
                    quantized_count += 1
            
            if quantized_count % 50 == 0 and quantized_count > 0:
                print(f"  已量化 {quantized_count} 个层...")
    
    print(f"[INFO] 共量化 {quantized_count} 个线性层")
    
    print(f"[INFO] 保存量化模型到: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    
    # 复制额外配置文件
    copy_processor_configs(model_path, output_path)
    
    # 保存量化配置
    quant_config = {
        "bits": bits,
        "group_size": group_size,
        "method": "manual_gptq",
        "model_type": "qwen3-vl",
    }
    import json
    with open(os.path.join(output_path, "quantize_config.json"), "w") as f:
        json.dump(quant_config, f, indent=2)
    
    print("[INFO] 量化完成!")
    return output_path


def copy_processor_configs(src_path: str, dst_path: str):
    """复制 VLM processor 配置文件"""
    config_files = [
        "preprocessor_config.json",
        "video_preprocessor_config.json", 
        "chat_template.json",
        "processor_config.json",
        "generation_config.json",
    ]
    
    for config_file in config_files:
        src = os.path.join(src_path, config_file)
        if os.path.exists(src):
            shutil.copy(src, dst_path)
            print(f"[INFO] 复制配置: {config_file}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL 模型量化工具")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="/home/fudan222/ct/Qwen3-VL/models/Qwen3-VL-8B",
        help="原始模型路径"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/fudan222/ct/Qwen3-VL/models/Qwen3-VL-8B-INT4",
        help="量化模型输出路径"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="autoround",
        choices=["autoround", "bitsandbytes", "manual"],
        help="量化方法: autoround (推荐), bitsandbytes, manual"
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="量化位宽"
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="量化组大小"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=128,
        help="校准样本数量"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Qwen3-VL 模型量化工具")
    print("=" * 60)
    print(f"方法: {args.method}")
    print(f"位宽: INT{args.bits}")
    print(f"组大小: {args.group_size}")
    print("=" * 60)
    
    if args.method == "autoround":
        quantize_with_autoround(
            model_path=args.model_path,
            output_path=args.output_path,
            bits=args.bits,
            group_size=args.group_size,
            num_samples=args.num_samples,
        )
    elif args.method == "bitsandbytes":
        quantize_with_bitsandbytes(
            model_path=args.model_path,
            output_path=args.output_path,
            bits=args.bits,
        )
    elif args.method == "manual":
        quantize_manual_gptq(
            model_path=args.model_path,
            output_path=args.output_path,
            bits=args.bits,
            group_size=args.group_size,
            num_samples=args.num_samples,
        )


if __name__ == "__main__":
    main()

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from auto_round import AutoRoundConfig
from PIL import Image
import time

model_path = '/home/fudan222/ct/Qwen3-VL/models/Qwen3-VL-8B-INT4'
print('[INFO] 加载量化模型...')

# 加载模型
quantization_config = AutoRoundConfig(backend='auto')
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map='auto',
    quantization_config=quantization_config
)
processor = AutoProcessor.from_pretrained(model_path)
print('[INFO] 模型加载成功!')

# 创建测试图像
print('[INFO] 创建测试图像...')
test_image = Image.new('RGB', (640, 480), color='lightblue')
from PIL import ImageDraw
draw = ImageDraw.Draw(test_image)
draw.rectangle([100, 100, 300, 300], fill='red', outline='darkred')
draw.rectangle([350, 150, 500, 350], fill='green', outline='darkgreen')
draw.ellipse([200, 250, 400, 400], fill='yellow', outline='orange')

# 准备消息
messages = [
    {
        'role': 'user',
        'content': [
            {'type': 'image', 'image': test_image},
            {'type': 'text', 'text': '请用中文描述这张图片中的内容。'}
        ]
    }
]

# 处理输入
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(
    text=[text],
    images=[test_image],
    return_tensors='pt',
    padding=True
)
inputs = inputs.to(model.device)

# 推理测试
print('[INFO] 开始推理测试...')
start_time = time.time()

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        temperature=None,
        top_p=None
    )

inference_time = time.time() - start_time

# 解码输出
generated_ids = output_ids[0, inputs['input_ids'].shape[1]:]
response = processor.decode(generated_ids, skip_special_tokens=True)

print(f'\\n[推理结果]')
print(f'回答: {response}')
print(f'推理时间: {inference_time:.2f} 秒')
print(f'生成token数: {len(generated_ids)}')
print(f'速度: {len(generated_ids)/inference_time:.1f} tokens/s')
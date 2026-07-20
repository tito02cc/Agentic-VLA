# IROS 2026 投稿指南

## 📁 文件结构

```
paper/
├── main.tex                    # 主论文 (IEEE双栏, 6+1页)
├── references.bib              # 参考文献 (28条)
├── supplementary.tex           # 补充材料
├── figures/
│   └── system_figure.tex       # TikZ系统架构图 (独立编译)
│
robot_grasp_rag/scripts/
├── run_honest_benchmark.py     # 诚实实验基准 (替代硬编码数据)
└── record_demo_video.py        # 补充视频录制脚本
```

---

## 🔧 投稿前必做步骤

### Step 1: 编译系统架构图

```bash
cd paper/figures
pdflatex system_figure.tex
# 输出: system_figure.pdf
# 然后在 main.tex 中取消注释:
# \includegraphics[width=\linewidth]{figures/system_overview.pdf}
```

如果需要修改图，编辑 `system_figure.tex` 中的 TikZ 代码。

### Step 2: 在GPU服务器上运行实验

**重要**: 以下实验需要在有 GPU 的服务器上运行 (推荐 RTX 5090/4090)。

```bash
# 1. 安装依赖
cd robot_grasp_rag
pip install -r requirements.txt

# 2. 快速测试 (验证环境, ~10分钟)
python scripts/run_honest_benchmark.py --mode quick --vlm-endpoint http://localhost:8000/v1

# 3. 完整实验 (生成论文数据, ~2-4小时)
python scripts/run_honest_benchmark.py --mode full --vlm-endpoint http://localhost:8000/v1

# 4. 消融实验
python scripts/run_honest_benchmark.py --mode ablation --vlm-endpoint http://localhost:8000/v1

# 5. 统计显著性测试 (5次重复, ~10-20小时)
python scripts/run_honest_benchmark.py --mode statistical --vlm-endpoint http://localhost:8000/v1 --repeats 5
```

**启动 VLM 服务** (在另一个终端):
```bash
python -m vllm.entrypoints.openai.api_server \
    --model models/Qwen3-VL-8B-INT4 \
    --served-model-name qwen3-vl-8b \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 4096
```

实验结果会自动保存到 `results/honest_benchmark/` 目录，包含:
- `benchmark_results.json` — 原始数据
- `comparison_table.tex` — 可直接插入论文的 LaTeX 表格
- `statistical_results.json` — 统计检验结果
- `video_frames/` — 实验过程帧 (用于补充材料)

### Step 3: 用实验数据更新论文

根据实际运行结果，更新 `main.tex` 中的以下表格:

| 表格 | 对应 Section | 数据来源 |
|------|-------------|---------|
| Table I (SOTA) | §IV-B | `--mode full` 输出 |
| Table II (统计) | §IV-B | `--mode statistical` 输出 |
| Table III (消融) | §IV-C | `--mode ablation` 输出 |
| Table IV (HAA-RAG) | §IV-D | `--mode full` 中 retrieval metrics |
| Table V (Scene Graph) | §IV-E | `--mode full` 中 scene graph results |
| Table VI (OOD) | §IV-F | `--mode full` 中 OOD results |
| Table VII (量化) | §IV-G | 已有真实数据 (`vlm_quant_ablation.json`) |

### Step 4: 录制补充视频

```bash
# 录制 MP4 视频 (需要 opencv-python)
python scripts/record_demo_video.py --output results/iros_demo.mp4

# 或者导出帧序列 (不需要 opencv)
python scripts/record_demo_video.py --output results/demo_frames --format frames
```

视频包含 5 个段落:
1. **Title Card** — 论文标题和作者信息
2. **System Overview** — 系统架构说明
3. **Grasp Demonstrations** — 三类任务的抓取演示
4. **Failure Recovery** — 自反思失败恢复演示
5. **Results Summary** — 定量结果展示

### Step 5: 编译论文

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# 编译补充材料
pdflatex supplementary.tex
```

---

## ⚠️ 关键注意事项

### 论文诚信
- **基线数据**: Table I 中传统方法 (GR-ConvNet, AnyGrasp, Contact-GraspNet, CLIPort) 的数据来源于各自论文的公开结果。在论文正文中已明确说明: *"Baseline performance data for traditional methods is obtained from their respective publications under comparable settings"*。
- **我们的方法**: "Ours" 和 "VLM-Only" 的数据必须来自 `run_honest_benchmark.py` 的实际运行结果。
- **统计检验**: 所有显著性声明必须基于多次独立重复 (`--mode statistical`)。

### 页面限制
- IROS 2026: **6 页正文 + 1 页参考文献**
- 当前 `main.tex` 约 6.5 页，可能需要微调表格大小或文字
- 使用 `\balance` 命令平衡最后一页的双栏

### 匿名审稿
- 作者信息已设为 "Anonymous IROS Submission"
- 录用后替换为真实作者信息 (模板已在注释中)
- 代码链接设为 `[anonymous]`

---

## 📊 当前数据可信度评估

| 实验 | 数据来源 | 可信度 | 备注 |
|------|---------|--------|------|
| VLM 量化 (Table VII) | RTX 5090 实测 | ✅ 真实 | `vlm_quant_ablation.json` |
| SOTA 对比 (Table I) | 需重新运行 | ⚠️ 待验证 | 用 `run_honest_benchmark.py` |
| 消融 (Table III) | 需重新运行 | ⚠️ 待验证 | 不能用 MockVLMClient |
| OOD (Table VI) | 需重新运行 | ⚠️ 待验证 | 不能硬编码 |
| HAA-RAG (Table IV) | 需重新运行 | ⚠️ 待验证 | 运行 retrieval evaluation |
| Scene Graph (Table V) | 需重新运行 | ⚠️ 待验证 | 运行 scene graph ablation |
| 统计显著性 (Table II) | 需运行 | ⚠️ 待验证 | 5次独立重复 |

---

## 🔗 提交清单

- [ ] 系统架构图编译为 PDF
- [ ] GPU 服务器上运行完整实验
- [ ] 用真实数据更新所有表格
- [ ] 统计显著性测试通过
- [ ] 录制补充视频
- [ ] 论文编译无错误
- [ ] 检查页数 ≤ 6+1
- [ ] 确认匿名化
- [ ] IEEE PDF eXpress 格式检查
- [ ] 补充材料提交 (视频 + supplementary.pdf)

---

## 🛠️ 常见问题

**Q: PyBullet 实验中成功率很低怎么办?**
A: 检查 VLM endpoint 是否正常响应。如果 VLM 不可用，系统会 fallback 到启发式方法，成功率约 40-50%。确保量化模型加载正确。

**Q: 实验结果与论文草稿数据差距太大?**
A: 这是预期的。原始草稿中的数据是模拟的，真实实验结果可能更低。根据实际结果诚实更新论文，调整 claim 的强度。如果某些 claim 不再成立，修改论述角度。

**Q: 如何加速实验?**
A: 使用 `--mode quick` 快速验证 (每个任务仅 3 次)。完整实验可用多 GPU 并行 (修改 `--vlm-endpoint` 指向不同端口)。

**Q: 编译 LaTeX 报错?**
A: 确保安装了完整的 TeX Live 发行版，特别是 `IEEEtran.cls`。运行: `tlmgr install IEEEtran`。

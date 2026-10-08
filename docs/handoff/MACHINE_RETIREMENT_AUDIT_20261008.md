# 旧机器停用前的资料核对（2026-10-08）

这是一份**项目资料快照**核对，不是整机镜像，也不保证换电脑后仿真环境可直接启动。旧目录名 `CARVE-VLA`，公开仓库名 `Agentic-VLA`；论文草稿中的方法名为 `BIND-VLA`。

## 已保存到哪里

| 位置 | 用途 | 核对状态 |
| --- | --- | --- |
| [GitHub: tito02cc/Agentic-VLA](https://github.com/tito02cc/Agentic-VLA) | 主项目代码、配置、测试、论文源稿/PDF、技术报告、汇总结果及精选视频；包含 Agentic RAG-VLM 的自有代码和论文 | 从独立发布目录提交；与原目录的核心源文件逐文件比较 |
| [私有 HF: Agentic-VLA-reproduction](https://huggingface.co/datasets/Minth-Group/Agentic-VLA-reproduction) | 先前选取的 LIBERO-PRO、RoboMME、RoboDojo、Agentic RAG-VLM/VLABench 一手证据 | 远端大小和 SHA-256 已核对；见[数据卡](HF_EVIDENCE_DATASET_CARD_20261008.md) |
| [私有 HF: Agentic-VLA-machine-handoff-20261008](https://huggingface.co/datasets/Minth-Group/Agentic-VLA-machine-handoff-20261008) | 完整 `results/`、`artifacts/`、Agentic RAG-VLM `output/`，交付物/视频、论文与自有中期材料、部分学校资产、小型 `weights/` 归一化文件、G2 非模型资料及第三方仓库本地补丁 | 7 个归档、3 个补丁均有远端条目；大文件远端 LFS SHA-256 与本机清单一致，小补丁已从远端下载复核；本机归档通过解压测试和 `sha256sum -c` |

私有 HF 中包含经用户授权**仅作私有备份**的个人与学校资料，不能因为 GitHub 仓库公开就把该数据集改为公开。备份完成时数据集提交为 `824a1d7c294869d5aaa91d7547c74c73fe7761d4`。如旧电脑之后继续产生新实验，这个快照不会自动更新。

## 在新电脑上恢复

仅写论文：克隆 GitHub 仓库，从 [`导航.md`](../../导航.md)进入技术报告、主稿、图表和公开示例即可。不要为了阅读文稿下载模型。

需要大规模一手结果或私有视频：先确认新电脑能登录 `Minth-Group` 的 HF 账号，再下载完整备份并验证：

```bash
git clone https://github.com/tito02cc/Agentic-VLA.git
hf auth login
hf download Minth-Group/Agentic-VLA-machine-handoff-20261008 \
  --repo-type dataset --local-dir ./Agentic-VLA-machine-handoff
cd Agentic-VLA-machine-handoff
sha256sum -c SHA256SUMS
mkdir -p extracted
tar -I zstd -xf archives/results-all.tar.zst -C extracted
```

其余归档按需解压到**空目录**，先查看内容再复制到 GitHub 克隆目录，避免覆盖后来修改过的文件。第三方补丁只在对应上游 commit 上应用；具体 base commit 和内容见私有 HF 数据卡。`SHA256SUMS` 验证传输完整性，不验证论文结论。

## 明确没有上传的内容

- 官方/第三方模型权重与大型缓存：PI0.5 各实验使用的不同 checkpoint、Qwen、RoboDojo、RoboMME、StarVLA 等。小型归一化 `weights/` 已私有备份，但**不是**模型主体。下载源与必要版本见 [`MODEL_AND_ENVIRONMENT_SETUP.md`](../../MODEL_AND_ENVIRONMENT_SETUP.md)。
- RoboDojo/Isaac 大型仿真资产、完整第三方仓库及虚拟环境、CUDA/显卡驱动等系统依赖。RoboDojo 资产原机约 66 GB；代码补丁不等于完整环境。
- 原项目 `.git` 历史与当时脏工作区本身。GitHub 是**选择性发布的可读快照**，不是该 31 GB 本地 `.git` 目录的镜像；不要假设所有旧分支、缓存对象和未纳入快照的改动都可恢复。
- `midterm/参考/`：含其他同学的个人参考材料，未因本项目用户授权就上传。原始 Agentic RAG-VLM 论文时期的私有 116 条经验库在备份前本机已不存在，不能从这次备份恢复。
- 除项目根目录和明确列出的 `/home/admin1/g2_multimodal_agent/` 非模型子目录以外的其他机器文件、登录令牌和密钥。本备份不是整机备份。

**停用旧电脑前最后一步：**在新电脑上实际完成一次 GitHub 克隆、HF 私有仓库登录、下载 `SHA256SUMS` 并检查论文 PDF/一条视频能打开；否则仍不能把“已上传”理解为“已可用”。

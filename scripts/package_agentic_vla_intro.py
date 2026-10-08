#!/usr/bin/env python3
"""Package the introduction and its directly cited evidence without changing sources."""

from __future__ import annotations

import hashlib
import io
import json
import posixpath
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
REPORT = Path("docs/reports/AGENTIC_VLA_TECHNICAL_REPORT.md")
INTRO = Path("deliverables/AGENTIC_VLA_TECHNICAL_INTRO")
PPT = INTRO / "Agentic_VLA_技术介绍.pptx"
PREFIX = "Agentic_VLA_介绍与交接材料"
OUTPUT = ROOT / "exports" / f"{PREFIX}.zip"

OPENING = """# Agentic VLA介绍与交接材料

证据截至2026-09-11。请先完整解压，再打开文档和PPT，不要直接在压缩包内播放。

## 从这里开始

1. [介绍PPT](deliverables/AGENTIC_VLA_TECHNICAL_INTRO/Agentic_VLA_技术介绍.pptx)：
   22页，正文1–20页，备份21–22页；第17页提供成功与失败视频的可点击链接。
2. [逐页讲解](deliverables/AGENTIC_VLA_TECHNICAL_INTRO/逐页讲解.md)：
   与PPT页码对应；PPT备注中也有讲解。
3. [完整技术底稿](docs/reports/AGENTIC_VLA_TECHNICAL_REPORT.md)：
   框架、关键技术、实验设置、正负结果、代码入口及证据边界。
4. [面试与汇报指南](docs/reports/FRAMEWORK_READINESS_AND_INTERVIEW_GUIDE.md)。
5. [Kiro完整交接提示词](docs/handoff/KIRO_CONTINUATION_PROMPT.md)：
   复制“提示词开始”至“提示词结束”的内容，并告知新电脑的实际仓库位置。
6. [框架图与中文说明](deliverables/AGENTIC_VLA_TECHNICAL_INTRO/figures/框架图说明.md)：
   同目录含独立PNG与PDF，与PPT第3页的原图一致。

## 视频

- [C3 layout 0成功](deliverables/AGENTIC_VLA_TECHNICAL_INTRO/media/robodojo_C3_layout0_success.mp4)
- [C3 layout 1失败](deliverables/AGENTIC_VLA_TECHNICAL_INTRO/media/robodojo_C3_layout1_failure.mp4)
- [B0 layout 0成功](artifacts/robodojo/recovery_lifecycle_20260910/B0/episode_0000000_cam_head_success.mp4)
- [C1 layout 0失败](artifacts/robodojo/recovery_lifecycle_20260910/C1/episode_0000000_cam_head_fail.mp4)

这些均为原始仿真视频，没有新增封面。PPT内链接为相对文件链接，不是内嵌视频；
请保留PPT旁的media文件夹。Office可能要求确认打开外部文件，也可直接播放上述MP4。
不同平台的PPT软件可能限制外部文件链接；链接已核对，未在各平台Office中逐一测试。
成功视频不等于已经证明Agent纠正了失败，解释以底稿与实验协议为准。

## 包含范围

保留仓库相对目录，包含技术底稿直接链接的文件：实验汇总、逐回合表、测试记录、
协议、关键代码与代表视频，以及PPT使用的框架图。MANIFEST.json记录各文件SHA-256。
没有修改原报告、PPT、视频或论文。

这是阅读与交接包，不是完整运行仓库。不包含模型权重、全部回合原始日志及视频、
虚拟环境、构建缓存或完整源码依赖；关键代码仅供阅读。底稿的直接Markdown文件链接
已检查，二级文档链接、JSON内部的绝对路径及代码依赖不保证离线齐全。
附带README中的.build重建路径属于原工作区，本包未包含该构建环境。
需要复现实验、运行审计脚本或继续开发时仍须完整项目及相应数据与模型。

本次仅整理已有材料，没有新增实验成绩。
"""


def direct_links() -> set[Path]:
    paths = set()
    for target in re.findall(r"\]\(([^)]+)\)", (ROOT / REPORT).read_text()):
        parsed = urlsplit(target.strip("<>"))
        if parsed.scheme or not parsed.path:
            continue
        source = (ROOT / REPORT.parent / unquote(parsed.path)).resolve()
        relative = source.relative_to(ROOT)
        if not source.is_file():
            raise FileNotFoundError(source)
        paths.add(relative)
    return paths


def build() -> None:
    files = direct_links() | {
        REPORT,
        PPT,
        INTRO / "README.md",
        INTRO / "逐页讲解.md",
        Path("docs/handoff/KIRO_CONTINUATION_PROMPT.md"),
        Path("paper/CARVE-VLA/figures/fig1_framework_fused.png"),
        Path(__file__).relative_to(ROOT),
    }
    files.update(path.relative_to(ROOT) for path in (ROOT / INTRO / "media").glob("*.mp4"))
    files.update(path.relative_to(ROOT) for path in (ROOT / INTRO / "figures").iterdir() if path.is_file())
    entries = {path.as_posix(): (ROOT / path).read_bytes() for path in sorted(files)}
    entries["打开说明.md"] = OPENING.encode("utf-8")
    manifest = {
        "evidence_as_of": "2026-09-11",
        "scope": "Presentation, report, notes, original videos, directly cited evidence; not a runnable repository.",
        "files": [
            {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in sorted(entries.items())
        ],
    }
    entries["MANIFEST.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp.zip")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(f"{PREFIX}/{name}", data)

    # Validate the exact archived PPT, not only its source on disk.
    with ZipFile(temporary) as archive:
        assert archive.testzip() is None
        for entry in manifest["files"]:
            data = archive.read(f"{PREFIX}/{entry['path']}")
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        videos = []
        with ZipFile(io.BytesIO(archive.read(f"{PREFIX}/{PPT.as_posix()}"))) as deck:
            for name in deck.namelist():
                if not name.startswith("ppt/slides/_rels/") or not name.endswith(".rels"):
                    continue
                for rel in ElementTree.fromstring(deck.read(name)):
                    target = unquote(rel.attrib.get("Target", ""))
                    if target.lower().endswith(".mp4"):
                        assert rel.attrib.get("TargetMode") == "External"
                        assert not urlsplit(target).scheme
                        linked = posixpath.normpath(f"{PPT.parent.as_posix()}/{target}")
                        assert f"{PREFIX}/{linked}" in archive.namelist(), linked
                        videos.append({"relationship": name, "target": target})
        assert len(videos) == 2, videos
        assert all(f"{PREFIX}/{path.as_posix()}" in archive.namelist() for path in direct_links())
    temporary.replace(OUTPUT)
    print(json.dumps({"archive": str(OUTPUT), "bytes": OUTPUT.stat().st_size,
                      "files": len(entries), "direct_report_targets": len(direct_links()),
                      "video_links": videos, "crc_and_sha256": "passed"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build()

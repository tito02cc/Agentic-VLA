#!/usr/bin/env python3
"""Build a portable report/video reading bundle; never alter experiment sources."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import posixpath
import re
import shutil
import subprocess
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from markdown_it import MarkdownIt


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/reports/AGENTIC_VLA_TECHNICAL_REPORT.md"
GUIDE = ROOT / "docs/reports/AGENTIC_VLA_STAGE_VIDEO_GUIDE.md"
PREFIX = "Agentic_VLA_技术报告与视频_20260918"
OUTPUT = ROOT / "deliverables" / PREFIX
ARCHIVE = ROOT / "exports" / f"{PREFIX}.zip"
LINK = re.compile(r"(!?)\[([^\]\n]*)\]\(([^)\n]+)\)")
CURRENT = ROOT / "artifacts/robodojo/organize_tool_return_validation_20260918"
DIAGNOSTIC = ROOT / "artifacts/robodojo/organize_target_rejection_20260918"
RUN = "2026-09-18_14-49-05_agent_on"

CSS = """
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;color:#202628;background:#fff;font:16px/1.85 'Noto Sans CJK SC','Microsoft YaHei',sans-serif;letter-spacing:0}
header{border-bottom:1px solid #d9dfe1;padding:14px 24px;background:#f5f7f7}
header a{margin-right:24px}a{color:#08645b;text-decoration:underline;text-underline-offset:3px;overflow-wrap:anywhere}
.layout{display:grid;grid-template-columns:245px minmax(0,1fr);max-width:1480px;margin:auto}
aside{padding:24px 18px;position:sticky;top:0;height:100vh;overflow:auto;border-right:1px solid #e0e4e5}
aside a{display:block;margin:10px 0;font-size:13px;text-decoration:none;line-height:1.65}
main{padding:30px 40px 90px;min-width:0;max-width:1110px}
h1{font-size:28px;line-height:1.5;margin:0 0 20px}h2{font-size:23px;line-height:1.5;margin:38px 0 16px;border-bottom:1px solid #d8dfe1;padding-bottom:8px}
h3{font-size:19px;line-height:1.6;margin-top:28px}h4{font-size:17px}
p,li,td,th{overflow-wrap:anywhere}p{margin:12px 0}li{margin:5px 0}
table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.7;margin:18px 0}
th,td{border:1px solid #dce2e3;padding:9px 12px;text-align:left;vertical-align:top}th{background:#f0f4f3;color:#17433d}
pre{background:#f5f6f7;border:1px solid #dce2e3;padding:15px;overflow-x:auto;font-size:13px;line-height:1.8}
code{font-family:'Noto Sans Mono CJK SC','DejaVu Sans Mono',monospace;font-size:.9em}
blockquote{margin:20px 0;padding:8px 20px;border-left:3px solid #547587;background:#f6f8f9}
img{max-width:100%;height:auto;display:block}video{display:block;width:100%;max-width:800px;aspect-ratio:4/3;background:#111;margin:16px 0}
.media-item{padding:20px 0 32px;border-bottom:1px solid #dce2e3}.media-item h2{border:0;margin-top:0}
.notice{padding:14px 18px;background:#f4f7f6;border-left:3px solid #2c7665}
@media(max-width:850px){.layout{display:block}aside{position:static;height:auto;border:0;border-bottom:1px solid #dce2e3}aside details:not([open]) a{display:none}main{padding:22px 16px 60px}h1{font-size:25px}table{font-size:12px}td,th{padding:6px}header{padding:12px 16px}}
@media print{header,aside{display:none}.layout{display:block}main{max-width:none;padding:0}h2,h3{break-after:avoid}tr,img,pre{break-inside:avoid}a{color:inherit}video{display:none}}
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def target_path(source: Path, target: str) -> Path | None:
    parsed = urlsplit(target.strip("<>"))
    if parsed.scheme or not parsed.path:
        return None
    path = (source.parent / unquote(parsed.path)).resolve()
    path.relative_to(ROOT)
    return path


def linked_paths(source: Path) -> set[Path]:
    return {path for match in LINK.finditer(source.read_text(encoding="utf-8"))
            if (path := target_path(source, match[3])) is not None}


def selected_videos() -> list[dict]:
    current = read_json(CURRENT / "run01/summary.json")
    native_root = ROOT / "artifacts/robodojo/sorting_development_20260915"
    native = read_json(native_root / "organize_native/summary.json")
    classify = read_json(native_root / "classify_native/summary.json")

    def head(summary):
        return next(Path(p) for p in summary["videos"] if "cam_head" in p)

    return [
        {"source": head(current), "name": "01_整理桌面_Agent_头部_75分_失败.mp4",
         "title": "最新整理桌面：Agent头部视角", "run": RUN, "score": 75,
         "description": "官方π0.5 + 4B VLM与受限Harness。1000步，整任务失败；约8秒处对应一次自主重新观察。不是已证明的恢复成功。"},
        {"source": head(native), "name": "02_整理桌面_原生VLA_头部_75分_失败.mp4",
         "title": "原生π0.5参照：整理桌面", "run": "2026-09-15_10-28-49_pi05_native", "score": 75,
         "description": "无Agent的历史原生回合。1000步、失败；与最新回合不是同步配对消融，不能据此证明收益或非劣。"},
        {"source": head(classify), "name": "03_语言分类_原生VLA_头部_0分_失败.mp4",
         "title": "第二任务：按语言指令分类", "run": "2026-09-15_10-38-05_pi05_native", "score": 0,
         "description": "无Agent。1100步、失败；说明任务内容和原生动作模型瓶颈，不能解释为新Harness失败。"},
        {"source": next(Path(p) for p in current["videos"] if "cam_left_wrist" in p),
         "name": "04_整理桌面_Agent_左腕_同回合.mp4", "title": "最新整理桌面：左腕辅助视角",
         "run": RUN, "score": 75, "description": "与视频01同一回合，查看局部操作和遮挡，不增加统计样本数。"},
        {"source": next(Path(p) for p in current["videos"] if "cam_right_wrist" in p),
         "name": "05_整理桌面_Agent_右腕_同回合.mp4", "title": "最新整理桌面：右腕辅助视角",
         "run": RUN, "score": 75, "description": "与视频01同一回合，查看另一侧局部观察，不增加统计样本数。"},
    ]


def rewrite_links(source: Path, dest: Path, mapping: dict[Path, Path], omitted: list) -> str:
    def replace(match):
        path = target_path(source, match[3])
        if path is None:
            return match[0]
        if path not in mapping:
            relative = path.relative_to(ROOT).as_posix()
            omitted.append({"document": str(dest), "source_target": relative})
            return f"{match[2]}（源仓库参考：`{relative}`，未随包附带）"
        target = posixpath.relpath(mapping[path].as_posix(), dest.parent.as_posix())
        fragment = urlsplit(match[3].strip("<>")).fragment
        if fragment:
            target += "#" + fragment
        return f"{match[1]}[{match[2]}](<{target}>)"
    return LINK.sub(replace, source.read_text(encoding="utf-8"))


def page(body: str, toc: list[tuple[str, str]], title: str, depth: int = 0) -> str:
    up = "../" * depth
    links = "".join(f'<a href="#{ident}">{html.escape(text)}</a>' for ident, text in toc)
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body>
<header><a href="{up}技术报告.html">完整技术报告</a><a href="{up}视频/index.html">展示视频</a><a href="{up}视频/观看说明.html">观看说明</a></header>
<div class="layout"><aside><details open><summary>目录</summary>{links}</details></aside><main>{body}</main></div>
</body></html>'''


def markdown_html(source: Path, title: str, depth: int = 0) -> str:
    text = source.read_text(encoding="utf-8")
    # Keep equations readable offline without a network-hosted math renderer.
    text = re.sub(r"\\\[(.*?)\\\]", lambda m: "\n```latex\n" + m[1].strip() + "\n```\n", text, flags=re.S)
    text = re.sub(r"\\\((.*?)\\\)", lambda m: "`" + m[1] + "`", text)
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = md.parse(text)
    toc = []
    for i, token in enumerate(tokens):
        if token.type == "heading_open":
            ident = f"section-{i}"
            token.attrSet("id", ident)
            if token.tag in {"h1", "h2"}:
                toc.append((ident, tokens[i + 1].content))
        for child in token.children or []:
            if child.type == "link_open":
                href = unquote(child.attrGet("href") or "")
                if href.endswith("技术报告.md") or href.endswith("观看说明.md"):
                    child.attrSet("href", quote(href[:-3] + ".html", safe="/"))
    return page(md.renderer.render(tokens, md.options, {}), toc, title, depth)


def verify_video(path: Path) -> dict:
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size:stream=codec_name,codec_type,width,height,avg_frame_rate,nb_frames",
        "-of", "json", str(path)], capture_output=True, text=True, check=True)
    metadata = json.loads(probe.stdout)
    decoded = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path),
        "-map", "0:v:0", "-f", "null", "-"], capture_output=True, text=True, check=True)
    if decoded.stderr.strip():
        raise RuntimeError(f"Decode diagnostics for {path}: {decoded.stderr}")
    stream = next(s for s in metadata["streams"] if s["codec_type"] == "video")
    frames = int(stream["nb_frames"])
    fps = float(Fraction(stream["avg_frame_rate"]))
    return {"file": str(path.relative_to(OUTPUT)), "sha256": digest(path),
            "codec": stream["codec_name"], "width": stream["width"], "height": stream["height"],
            "frames": frames, "fps": fps, "duration_seconds": float(metadata["format"]["duration"]),
            "full_decode": "passed", "byte_preserving_copy": True}


def check_markdown_links(root: Path) -> int:
    count = 0
    for source in root.rglob("*.md"):
        for match in LINK.finditer(source.read_text(encoding="utf-8")):
            parsed = urlsplit(match[3].strip("<>"))
            if parsed.scheme or not parsed.path:
                continue
            target = (source.parent / unquote(parsed.path)).resolve()
            target.relative_to(root.resolve())
            if not target.is_file():
                raise FileNotFoundError(f"{source}: {target}")
            count += 1
    return count


def build(replace: bool) -> None:
    if OUTPUT.exists() and not replace:
        raise FileExistsError(f"Use --replace for this generated output only: {OUTPUT}")
    if OUTPUT.exists():
        # Only this script's own, explicitly named generated directory is replaceable.
        if not (OUTPUT / "MANIFEST.json").is_file():
            raise RuntimeError("Refusing to replace an output without its manifest")
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    videos = selected_videos()
    mapping = {REPORT: Path("技术报告.md"), GUIDE: Path("视频/观看说明.md")}
    for v in videos:
        mapping[v["source"].resolve()] = Path("视频") / v["name"]
    sources = linked_paths(REPORT) | linked_paths(GUIDE) | {REPORT, GUIDE}
    extras = [CURRENT / "PLAN.md", CURRENT / "audit_metrics.json", CURRENT / "video_metadata.json",
              CURRENT / "tests.xml", CURRENT / "head_frame_201.png", CURRENT / "head_frame_1000.png",
              ROOT / "artifacts/robodojo/sorting_development_20260915/organize_native/summary.json",
              ROOT / "artifacts/robodojo/sorting_development_20260915/classify_native/summary.json",
              ROOT / "configs/robodojo_sorting_agent.json"]
    sources.update(extras)
    sources.update(p for p in (CURRENT / "run01").rglob("*") if p.suffix in {".json", ".jsonl"})
    sources.update(p for p in DIAGNOSTIC.rglob("*") if p.suffix in {".json", ".jsonl", ".png", ".md", ".xml"})
    historical = 0
    for source in sorted(sources):
        if not source.is_file():
            raise FileNotFoundError(source)
        if source in mapping:
            continue
        relative = source.relative_to(ROOT)
        if source.suffix == ".mp4":
            historical += 1
            mapping[source] = Path("视频/历史参考") / f"{historical:02d}_{source.parent.name}_{source.name}"
        else:
            mapping[source] = Path("核查资料") / relative

    omitted = []
    provenance = {}
    for source, destination in sorted(mapping.items()):
        dest = OUTPUT / destination
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".md":
            dest.write_text(rewrite_links(source, destination, mapping, omitted), encoding="utf-8")
            transformation = "portable_links_only"
        else:
            shutil.copy2(source, dest)
            transformation = "byte_identical"
            assert digest(source) == digest(dest)
        provenance[destination.as_posix()] = {
            "source": source.relative_to(ROOT).as_posix(), "source_sha256": digest(source),
            "transformation": transformation}

    report_html = OUTPUT / "技术报告.html"
    report_html.write_text(markdown_html(OUTPUT / "技术报告.md", "Agentic VLA 完整技术报告"), encoding="utf-8")
    (OUTPUT / "视频/观看说明.html").write_text(
        markdown_html(OUTPUT / "视频/观看说明.md", "实验视频观看说明", 1), encoding="utf-8")
    video_checks = [verify_video(path) for path in sorted((OUTPUT / "视频").rglob("*.mp4"))]
    checks_by_name = {c["file"]: c for c in video_checks}
    body = '<h1>Agentic VLA 阶段展示视频</h1><p>证据截至2026-09-18；原始仿真录像，无封面、剪辑或变速。</p>'
    body += '<p class="notice">主展示五段对应三个回合，全部整任务失败。75分不是75%成功率；25FPS录像不代表包含模型等待的实时速度。最近视觉诊断没有新增机器人回合。</p>'
    toc = []
    for i, v in enumerate(videos, 1):
        c = checks_by_name[(Path("视频") / v["name"]).as_posix()]
        ident = f"video-{i}"
        toc.append((ident, v["title"]))
        body += f'''<section class="media-item" id="{ident}"><h2>{i}. {html.escape(v['title'])}</h2>
<p>{html.escape(v['description'])}</p><p>{c['width']} × {c['height']} · {c['frames']}帧 · {c['duration_seconds']:.2f}秒 · 全片解码通过</p>
<video controls preload="metadata" playsinline src="{quote(v['name'])}"></video>
<p><a href="{quote(v['name'])}">打开原始MP4</a> · <a href="观看说明.html">实验与时间点说明</a></p></section>'''
    body += '<h2 id="history">历史参考，不作为当前效果证明</h2><ul>'
    toc.append(("history", "历史参考"))
    for path in sorted((OUTPUT / "视频/历史参考").glob("*.mp4")):
        relative = path.relative_to(OUTPUT / "视频").as_posix()
        body += f'<li><a href="{quote(relative)}">{html.escape(path.name)}</a></li>'
    body += '</ul>'
    (OUTPUT / "视频/index.html").write_text(page(body, toc, "阶段实验视频", 1), encoding="utf-8")

    readme = """# Agentic VLA 技术报告与视频

证据截至2026-09-18。先完整解压，再打开文件；不需要原工作站或网络。

## 两个主要入口

1. [完整技术报告HTML](技术报告.html)：带章节目录，可直接用浏览器阅读。
   [Markdown底稿](技术报告.md)供编辑、AI阅读；内容相同，HTML中的公式保留LaTeX形式。
2. [展示视频](视频/index.html)：五段当前主展示录像、五段历史补充录像；
   [观看说明](视频/观看说明.html)给出任务、模型、评分、时间点和讲解边界。

最新整理桌面回合为75/100、整任务失败，真实验证一次自主重新观察后的工具返回链。
其后视觉候选未通过准入，没有新的机器人回合。历史推理加速属于旧PyTorch配置，
不能与最新JAX模型成绩混为一套效果。此包是阶段汇报材料，不是实验完成或录用证明。

## 包含与不包含

- 视频均为原始文件逐字节复制，不加封面、剪辑、加速或人工动作；三路当前相机是同一回合。
- 核查资料保存主报告直接引用的证据、必要代码摘录、最新回合日志和最新视觉诊断原始输入/响应。
- 原始JSON/JSONL未改写，其中绝对路径是来源记录，不保证在另一台电脑可运行。
- Markdown仅适配本地链接；未随包附带的二级资料明确标为源仓库路径，避免形成打不开的链接。
- 旧计划和旧说明为历史证据，以根目录报告的最新状态为准。本包不包含旧PPT、模型权重、虚拟环境或完整源码依赖。
- 没有新增GPU实验，没有修改或删除paper目录，也没有覆盖历史实验结局。

## 文件核验

`MANIFEST.json`记录每个内容文件的SHA-256、来源与是否调整文档链接；
`文件校验.sha256`可在本目录运行 `sha256sum -c 文件校验.sha256` 检查。
`核查结果.json`记录视频逐帧解码、尺寸、帧数和本地链接检查；归档还经过ZIP CRC检查。
这只证明材料完整可读取，不把解码检查等同于人工逐帧语义标注。
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    linked_count = check_markdown_links(OUTPUT)
    dump(OUTPUT / "核查结果.json", {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_as_of": "2026-09-18", "new_gpu_experiments": False,
        "markdown_links_checked": linked_count, "missing_clickable_local_markdown_links": 0,
        "secondary_references_not_packaged": omitted,
        "videos": video_checks, "primary_video_count": len(videos),
        "primary_unique_episodes": len({v["run"] for v in videos}),
        "visual_review_scope": "Original current head frames 201/1000 and existing timeline sheets; not every frame human-annotated.",
    })
    files = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file():
            name = path.relative_to(OUTPUT).as_posix()
            files.append({"path": name, "bytes": path.stat().st_size, "sha256": digest(path),
                          **provenance.get(name, {"transformation": "generated_reading_material"})})
    dump(OUTPUT / "MANIFEST.json", {"evidence_as_of": "2026-09-18", "files": files})
    checksums = [(f["sha256"], f["path"]) for f in files]
    checksums.append((digest(OUTPUT / "MANIFEST.json"), "MANIFEST.json"))
    (OUTPUT / "文件校验.sha256").write_text(
        "".join(f"{sha}  {name}\n" for sha, name in checksums), encoding="utf-8")

    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    temp = ARCHIVE.with_suffix(".tmp.zip")
    with ZipFile(temp, "w", ZIP_DEFLATED) as archive:
        for path in sorted(OUTPUT.rglob("*")):
            if path.is_file():
                archive.write(path, Path(PREFIX) / path.relative_to(OUTPUT),
                              compress_type=ZIP_STORED if path.suffix == ".mp4" else ZIP_DEFLATED)
    with ZipFile(temp) as archive:
        assert archive.testzip() is None
        for entry in files:
            data = archive.read(f"{PREFIX}/{entry['path']}")
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    temp.replace(ARCHIVE)
    print(json.dumps({"archive": str(ARCHIVE), "bytes": ARCHIVE.stat().st_size,
                      "output_directory": str(OUTPUT), "content_files": len(files),
                      "videos": len(video_checks), "primary_videos": len(videos),
                      "local_markdown_links_checked": linked_count, "zip_crc_sha256": "passed"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true", help="Rebuild only this script's generated bundle")
    build(parser.parse_args().replace)

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _sample_frames(video_path: Path, num_frames: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Failed to read frame count from: {video_path}")

    frames = []
    select = np.linspace(0, max(frame_count - 1, 0), num_frames, dtype=int)
    for idx in select:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames could be extracted from {video_path}")
    return frames


def _to_pil(frame, height: int):
    img = Image.fromarray(np.asarray(frame).astype(np.uint8))
    width = int(img.width * (height / img.height))
    return img.resize((width, height))


def _load_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def build_contact_sheet(failure_video: Path, success_video: Path, output_path: Path, title: str, num_frames: int):
    failure_frames = _sample_frames(failure_video, num_frames)
    success_frames = _sample_frames(success_video, num_frames)

    font_title = _load_font(26)
    font_label = _load_font(20)
    font_small = _load_font(16)

    target_h = 180
    fail_imgs = [_to_pil(frame, target_h) for frame in failure_frames]
    succ_imgs = [_to_pil(frame, target_h) for frame in success_frames]

    cell_w = max(max(img.width for img in fail_imgs), max(img.width for img in succ_imgs))
    pad = 18
    left_label_w = 120
    title_h = 60
    row_label_h = 34

    total_w = left_label_w + pad + num_frames * (cell_w + pad) + pad
    total_h = title_h + 2 * (row_label_h + target_h + pad) + pad
    canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((pad, 14), title, fill=(17, 24, 39), font=font_title)
    draw.text((pad, 72), "Failure case", fill=(185, 28, 28), font=font_label)
    draw.text((pad, 72 + row_label_h + target_h + pad), "Success case", fill=(29, 78, 216), font=font_label)

    for row_idx, imgs in enumerate([fail_imgs, succ_imgs]):
        y0 = title_h + row_idx * (row_label_h + target_h + pad)
        draw.text((left_label_w + pad, y0 - 4), "Early          Mid          Late", fill=(75, 85, 99), font=font_small)
        for col_idx, img in enumerate(imgs):
            x = left_label_w + pad + col_idx * (cell_w + pad)
            y = y0 + row_label_h
            canvas.paste(img, (x, y))
            draw.rectangle([x, y, x + img.width, y + img.height], outline=(209, 213, 219), width=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare a qualitative comparison figure from rollout videos.")
    parser.add_argument("--failure-video", required=True)
    parser.add_argument("--success-video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Qualitative comparison of rollout trajectories")
    parser.add_argument("--num-frames", type=int, default=4)
    args = parser.parse_args()

    build_contact_sheet(
        Path(args.failure_video),
        Path(args.success_video),
        Path(args.output),
        args.title,
        args.num_frames,
    )
    print(args.output)


if __name__ == "__main__":
    main()

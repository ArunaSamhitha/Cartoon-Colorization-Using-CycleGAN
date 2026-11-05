"""Pull every Nth frame out of one or more cartoon videos into a folder.

    python scripts/extract_frames.py --videos clips/ --out data/frames --every 12
"""
import argparse
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", required=True, help="A video file or a folder of videos")
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=int, default=12, help="Keep one frame out of every N")
    args = p.parse_args()

    src, out = Path(args.videos), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    videos = [src] if src.is_file() else sorted(v for v in src.iterdir() if v.suffix.lower() in VIDEO_EXTS)
    saved = 0
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % args.every == 0:
                cv2.imwrite(str(out / f"{video.stem}_{i:06d}.jpg"), frame)
                saved += 1
            i += 1
        cap.release()
    print(f"Saved {saved} frames from {len(videos)} video(s) to {out}")


if __name__ == "__main__":
    main()

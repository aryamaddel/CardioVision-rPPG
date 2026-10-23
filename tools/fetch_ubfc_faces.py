"""Download the open UBFC-rPPG-Faces mirror (Hugging Face, Horusprg).

316 MB zip holding 42 subjects: ~1,500-2,050 x 128x128 mpeg4 clip files
(each a 5-frame sliding window, stride 1) + metadata.csv with one PPG
sample per clip. Download with resume support.

Usage:
  python tools/fetch_ubfc_faces.py [dest]
  dest defaults to TMP / "UBFC-rPPG-Faces.zip" (resumes a partial file).
"""
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
URL = ("https://huggingface.co/datasets/Horusprg/UBFC-rPPG-Faces/"
       "resolve/main/UBFC-rPPG-Faces.zip")
TMP = ROOT / "tmp"


def fetch(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = int(requests.head(URL, allow_redirects=True).headers["content-length"])
    have = dest.stat().st_size if dest.exists() else 0
    if have == total:
        print(f"already complete: {dest} ({total} bytes)")
        return
    print(f"{dest}  {have/1e6:.0f} / {total/1e6:.0f} MB")
    t0 = time.time()
    while have < total:
        with requests.get(URL, stream=True,
                          headers={"Range": f"bytes={have}-"}) as r:
            r.raise_for_status()
            with open(dest, "ab") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    have += len(chunk)
                    if have % 52428800 < 1048576:
                        dt = time.time() - t0
                        print(f"  {have/1e6:8.0f} MB  {have/1e6/max(dt,1e-6):.1f} MB/s",
                              flush=True)
        if time.time() - t0 > 7200:
            print("giving up after 2 h")
            return
    print(f"done: {dest} ({dest.stat().st_size} bytes, "
          f"{(dest.stat().st_size == total) and 'size OK' or 'SIZE MISMATCH'})")


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else TMP / "UBFC-rPPG-Faces.zip"
    fetch(dest)

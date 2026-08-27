#!/usr/bin/env python3
"""GPT-Image-1 edits of REAL photos (ft6).

The persistent misses in the field are photoreal "snapshot" fakes and AI
edits of real photos. UCSC-VLAA/GPT-Image-Edit-1.5M re-generates the outputs
of existing editing datasets with GPT-Image-1; its UltraEdit subset has REAL
input photos, so every sample is a (real photo, GPT-edited photo) pair of
the same scene — the cleanest possible teaching signal for the generator
fingerprint. (The HQ-Edit subset's inputs are DALL-E synthetic; not used.)

The data ships as split tar.gz parts (~21GB each). A split gzip stream is
readable as a prefix, so this streams parts sequentially and stops at the
caps — no multi-hundred-GB download.

Writes gptedit_* (label 1) and editsrc_* (label 0) into DATA/cartoons, with
the usual contiguous tail heldout; manifests() picks them up by prefix.
"""
import io
import os
import shutil
import tarfile
import time

import requests

from materialize_eval import DATA
from materialize_cartoons import manifests

BASE = ("https://huggingface.co/datasets/UCSC-VLAA/GPT-Image-Edit-1.5M/resolve/main/"
        "gpt-edit/ultraedit.tar.gz.part{:03d}")
PARTS = 4  # all of ultraedit; the loop stops early once caps are met
CAPS = {"gptedit": (12000, 1200), "editsrc": (8000, 800)}
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


class Chain(io.RawIOBase):
    """Sequential byte stream over several HTTP parts, resuming a dropped
    connection with a Range request (the CDN cuts long streams mid-part)."""

    RETRIES = 8

    def __init__(self, urls):
        self.urls, self.it, self.buf = list(urls), None, b""
        self.url, self.offset = None, 0

    def readable(self):
        return True

    def _open(self, url, offset):
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        r = requests.get(url, stream=True, timeout=120, headers=headers)
        r.raise_for_status()
        if offset and r.status_code != 206:
            raise RuntimeError("server ignored Range; cannot resume")
        self.url, self.offset = url, offset
        self.it = r.iter_content(chunk_size=1 << 20)

    def _open_next(self):
        if not self.urls:
            return False
        url = self.urls.pop(0)
        print(f"  streaming {url.rsplit('/', 1)[-1]}", flush=True)
        self._open(url, 0)
        return True

    def _chunk(self):
        for attempt in range(self.RETRIES):
            try:
                return next(self.it)
            except StopIteration:
                return None
            except (requests.RequestException, ConnectionError) as e:
                print(f"  connection dropped at {self.offset / 1e9:.2f} GB ({type(e).__name__}); resuming", flush=True)
                time.sleep(2 * (attempt + 1))
                self._open(self.url, self.offset)
        raise RuntimeError("too many resume attempts")

    def readinto(self, b):
        while not self.buf:
            if self.it is None and not self._open_next():
                return 0
            chunk = self._chunk()
            if chunk is None:
                self.it = None
                continue
            self.buf = chunk
            self.offset += len(chunk)
        n = min(len(b), len(self.buf))
        b[:n] = self.buf[:n]
        self.buf = self.buf[n:]
        return n


def main():
    marker = f"{DATA}/cartoons/.edits_done"
    if os.path.exists(marker):
        print("edits: done marker present, skipping")
        return
    os.makedirs(f"{DATA}/cartoons/train", exist_ok=True)
    os.makedirs(f"{DATA}/cartoons/heldout", exist_ok=True)
    written = {k: [] for k in CAPS}
    total = {k: sum(v) for k, v in CAPS.items()}
    stream = io.BufferedReader(Chain(BASE.format(i) for i in range(1, PARTS + 1)), 1 << 22)
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as tf:
            for m in tf:
                if all(len(written[k]) >= total[k] for k in CAPS):
                    break
                if not m.isfile():
                    continue
                name = m.name.lower()
                if not name.endswith(IMG_EXT):
                    continue
                kind = "gptedit" if "output" in name else "editsrc" if "input" in name else None
                if kind is None or len(written[kind]) >= total[kind]:
                    continue
                b = tf.extractfile(m).read()
                p = f"{DATA}/cartoons/train/{kind}_{len(written[kind]):06d}{os.path.splitext(name)[1]}"
                with open(p, "wb") as f:
                    f.write(b)
                written[kind].append(p)
                n = sum(len(v) for v in written.values())
                if n % 1000 == 0:
                    print(f"  {n}: " + " ".join(f"{k}={len(v)}" for k, v in written.items()), flush=True)
    except (EOFError, tarfile.ReadError) as e:
        print(f"  stream ended: {e}")
    for kind, paths in written.items():
        ho = min(CAPS[kind][1], len(paths) // 5)
        for p in paths[-ho:] if ho else []:
            shutil.move(p, f"{DATA}/cartoons/heldout/{os.path.basename(p)}")
        print(f"{kind}: {len(paths) - ho} train / {ho} heldout")
    open(marker, "w").close()
    manifests()


if __name__ == "__main__":
    main()

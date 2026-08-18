#!/usr/bin/env python3
"""One-off: remove AppleDouble resource-fork junk that rode in via the
macOS-built synth tar. The files were renamed synth_NNNNN.* on ingest, so the
dot-prefix filter in manifests() can't catch them — detect by magic bytes
instead. Re-run materialize_screenshots.py afterwards to rebuild manifests.

Usage: python3 cleanup_appledouble.py [dir ...]   (default: screenshots dirs)
"""
import os
import sys

DATA = os.path.expanduser(os.environ.get("DATA", "~/data"))

dirs = sys.argv[1:] or [f"{DATA}/screenshots/train", f"{DATA}/screenshots/heldout"]
n = 0
for d in dirs:
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        if not os.path.isfile(p):
            continue
        with open(p, "rb") as fh:
            head = fh.read(4)
        if head == b"\x00\x05\x16\x07":
            os.remove(p)
            n += 1
print(f"removed {n} AppleDouble files")

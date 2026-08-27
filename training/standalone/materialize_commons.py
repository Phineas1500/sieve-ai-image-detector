#!/usr/bin/env python3
"""Press / event photography hard reals (ft6).

Field false positives cluster on professional news-wire portraits: telephoto,
shallow depth of field, clean skin, public figures at podiums. Wikimedia
Commons' "Photographs by Gage Skidmore" (CC BY-SA, ~121k files) is exactly
that distribution — political events, conventions, red carpets. Strided
sample across the whole category, served as 1280px JPEG thumbnails.

Writes press_* (label 0) into DATA/cartoons with the usual tail heldout.
"""
import concurrent.futures as cf
import os
import shutil
import time

import requests

from materialize_eval import DATA
from materialize_cartoons import manifests

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "SieveDataBot/1.0 (https://github.com/Phineas1500/sieve-ai-image-detector; sriram.kiron@gmail.com)"}
CAT = "Category:Photographs by Gage Skidmore"
STRIDE, TRAIN, HELDOUT, WIDTH = 25, 4500, 450, 1280


def titles():
    cont = {}
    while True:
        r = requests.get(API, params=dict(action="query", list="categorymembers", cmtitle=CAT,
                                          cmtype="file", cmlimit=500, format="json", **cont),
                         headers=UA, timeout=60)
        r.raise_for_status()
        j = r.json()
        for m in j["query"]["categorymembers"]:
            if "|" not in m["title"]:
                yield m["title"]
        if "continue" not in j:
            return
        cont = j["continue"]
        time.sleep(0.1)


def thumbs(batch):
    r = requests.get(API, params=dict(action="query", titles="|".join(batch), prop="imageinfo",
                                      iiprop="url|mime", iiurlwidth=WIDTH, format="json"),
                     headers=UA, timeout=60)
    r.raise_for_status()
    out = []
    for p in r.json().get("query", {}).get("pages", {}).values():
        ii = (p.get("imageinfo") or [None])[0]
        if ii and ii.get("mime") == "image/jpeg" and ii.get("thumburl"):
            out.append(ii["thumburl"])
    return out


def fetch(url):
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 200 and r.content[:2] == b"\xff\xd8":
                return r.content
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
        except requests.RequestException:
            time.sleep(2)
    return None


def main():
    marker = f"{DATA}/cartoons/.press_done"
    if os.path.exists(marker):
        print("press: done marker present, skipping")
        return
    os.makedirs(f"{DATA}/cartoons/train", exist_ok=True)
    os.makedirs(f"{DATA}/cartoons/heldout", exist_ok=True)
    want = TRAIN + HELDOUT
    picked = [t for i, t in enumerate(titles()) if i % STRIDE == 0]
    print(f"  {len(picked)} titles sampled (stride {STRIDE})")
    urls = []
    for i in range(0, len(picked), 50):
        urls.extend(thumbs(picked[i:i + 50]))
        time.sleep(0.2)
        if len(urls) >= want:
            break
    print(f"  {len(urls)} thumbnail urls")
    written = []
    with cf.ThreadPoolExecutor(4) as ex:
        for b in ex.map(fetch, urls[:want]):
            if not b:
                continue
            p = f"{DATA}/cartoons/train/press_{len(written):06d}.jpg"
            with open(p, "wb") as f:
                f.write(b)
            written.append(p)
            if len(written) % 500 == 0:
                print(f"  {len(written)} downloaded", flush=True)
    ho = min(HELDOUT, len(written) // 5)
    for p in written[-ho:] if ho else []:
        shutil.move(p, f"{DATA}/cartoons/heldout/{os.path.basename(p)}")
    open(marker, "w").close()
    print(f"press: {len(written) - ho} train / {ho} heldout")
    manifests()


if __name__ == "__main__":
    main()

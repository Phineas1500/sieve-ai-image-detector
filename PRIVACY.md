# Sieve — Privacy Policy

**Effective: August 19, 2026**

Sieve is built so that it *cannot* see your data.

## What Sieve collects

Nothing. Sieve has no servers, no analytics, no telemetry, no accounts, and no
tracking of any kind.

## How images are processed

All image analysis runs entirely on your device, inside your browser, using a
local machine-learning model (ONNX Runtime Web on WebGPU or WASM). Images you
browse are fetched by the extension only to be decoded and scored locally —
they are never transmitted anywhere.

## Network access

Sieve makes exactly one kind of network request of its own: a **one-time,
user-initiated download** of its detection model (~45 MB) from a pinned,
checksum-verified GitHub release URL. This happens only when you press
"Download model" during setup or after a model update, and the download
contains no identifying information beyond a standard HTTP request. After
that, Sieve operates fully offline.

## Data storage

Settings (threshold, blur preference) are stored in Chrome's extension storage
on your device. The downloaded model and recent analysis scores are stored
locally in your browser. Nothing is synced to or shared with any third party
by Sieve.

## Reporting misclassifications

If you choose to report a misclassified image, Sieve opens a pre-filled GitHub
issue **in your browser for you to review and submit yourself**. The extension
itself sends nothing; you control exactly what is posted, publicly, on GitHub.

## Changes

Any change to this policy will appear in this file's git history —
transparency by version control.

## Contact

Open an issue at https://github.com/Phineas1500/sieve-ai-image-detector/issues

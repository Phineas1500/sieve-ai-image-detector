#!/usr/bin/env python3
"""Standalone fp16 ONNX export (native torch half, fp32 I/O boundary) —
port of the Modal export_onnx_fp16. Usage:
  python3 -u export_fp16.py --ckpt ~/data/ckpts/ft2/best.pt --out ~/ft2_best_fp16.onnx
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-size", type=int, default=384)
    args = ap.parse_args()

    import numpy as np
    import torch

    from vendor.cf_models import ViTClassifier

    C = args.input_size
    model = ViTClassifier(model_size="small", input_size=C, patch_size=16, device="cpu")
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd.get("model_state_dict", sd))
    model.eval().cuda()

    x = torch.randn(4, 3, C, C, device="cuda")
    with torch.no_grad():
        ref = model(x).squeeze(-1).float().cpu().numpy()

    model.half()

    class Fp32Boundary(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, t):
            return self.m(t.half()).float()

    wrapped = Fp32Boundary(model).eval().cuda()
    out = os.path.expanduser(args.out)
    kwargs = dict(
        input_names=["input"], output_names=["logit"],
        dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
    )
    try:
        # newer torch defaults to the dynamo exporter, which externalizes the
        # weights into a sidecar .data file — the extension needs ONE file
        torch.onnx.export(wrapped, x, out, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(wrapped, x, out, **kwargs)

    import onnxruntime as onnxrt

    sess = onnxrt.InferenceSession(out, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": x.cpu().numpy()})[0].squeeze(-1).astype(np.float32)
    err = float(np.abs(ref - got).max())
    print(f"fp16 parity max logit err: {err:.4f}; size {os.path.getsize(out)/1e6:.1f}MB")
    assert err < 0.05, "parity too loose"


if __name__ == "__main__":
    main()

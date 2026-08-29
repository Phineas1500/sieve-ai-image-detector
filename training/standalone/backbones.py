"""Backbone registry for the detector head.

"cf" is the Community Forensics ViT-S/16 @384 we have shipped since v0.1 —
a forensic-pretrained small ViT. The others are large self-supervised /
contrastive ViTs (DINOv2, CLIP) whose frozen features are known to
generalize to unseen generators far better than a small fully fine-tuned
ViT (the UnivFD line of work); we fine-tune only the last few blocks + head
(`freeze_blocks`) to keep that property. Checkpoints record their backbone
so eval/export rebuild the right graph.
"""
import torch
import torch.nn as nn

SPECS = {
    # name: (timm model, native input, resize_shorter_side)
    "cf": (None, 384, 440),
    "dinov2_b14": ("vit_base_patch14_dinov2.lvd142m", 224, 256),
    "dinov2_b14_336": ("vit_base_patch14_dinov2.lvd142m", 336, 384),
    "clip_b16": ("vit_base_patch16_clip_224.openai", 224, 256),
    "dinov2_s14": ("vit_small_patch14_dinov2.lvd142m", 224, 256),
}


class TimmClassifier(nn.Module):
    def __init__(self, timm_name, input_size, pretrained=True):
        super().__init__()
        import timm

        self.vit = timm.create_model(timm_name, pretrained=pretrained, num_classes=1, img_size=input_size)

    def forward(self, x):
        return self.vit(x)


def sizes(name):
    _, c, r = SPECS[name]
    return c, r


def build(name, hf_repo=None, pretrained=True):
    """Return an nn.Module with a 1-logit output for backbone `name`."""
    if name == "cf":
        from vendor.cf_models import ViTClassifier

        if hf_repo and pretrained:
            return ViTClassifier.from_pretrained(hf_repo, device="cpu")
        return ViTClassifier(model_size="small", input_size=384, patch_size=16, device="cpu")
    timm_name, c, _ = SPECS[name]
    return TimmClassifier(timm_name, c, pretrained=pretrained)


def freeze_blocks(model, n):
    """Freeze embeddings and the first `n` transformer blocks; train the rest."""
    if n <= 0:
        return 0
    frozen = 0
    for pname, p in model.named_parameters():
        parts = pname.split(".")
        if "blocks" in parts:
            idx = int(parts[parts.index("blocks") + 1])
            trainable = idx >= n
        else:
            trainable = any(k in pname for k in ("head", "norm", "fc_norm"))
        p.requires_grad = trainable
        frozen += 0 if trainable else 1
    return frozen

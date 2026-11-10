"""Build CircuitFormer from the third_party clone without editing it, attach LoRA.

Run a synthetic end-to-end check:  python -m lora_router.build [path/to/circuitformer.ckpt]
"""
import sys
from pathlib import Path
from unittest import mock

import torch

from .lora import apply_lora

CF_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "circuitformer"


def build_circuitformer(ckpt_path=None):
    """Return CircuitFormer, with the released weights if `ckpt_path` is given."""
    if str(CF_ROOT) not in sys.path:
        sys.path.insert(0, str(CF_ROOT))
    import numpy as np
    if not hasattr(np, "float"):  # their metrics.py uses the alias numpy >= 1.24 removed
        np.float = float
    from model.circuitformer import CircuitFormer

    # The constructor reads an ImageNet ResNet18 from a hard-coded relative path.
    # Skip it: the full checkpoint overwrites those weights anyway.
    with mock.patch("torch.load", return_value={"conv1.weight": None}):
        model = CircuitFormer()

    if ckpt_path:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        state = {k.removeprefix("model."): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    return model


def synthetic_batch(n_designs=2, n_cells=2000, device="cuda"):
    """Random cell rectangles in the collate format: [x1, y1, x2, y2, offsets]."""
    n = n_designs * n_cells
    x1, y1 = torch.rand(n, device=device) * 900, torch.rand(n, device=device) * 900
    x2, y2 = x1 + 1 + torch.rand(n, device=device) * 20, y1 + 1 + torch.rand(n, device=device) * 20
    offsets = torch.arange(1, n_designs + 1, device=device, dtype=torch.int32) * n_cells
    return [x1, y1, x2, y2, offsets]


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    torch.manual_seed(0)
    model = build_circuitformer(ckpt).cuda().eval()
    batch = synthetic_batch()
    fresh = lambda: [t.clone() for t in batch]  # the model mutates its input list

    with torch.no_grad():
        before = model(fresh())
    trainable, total = apply_lora(model, rank=8, alpha=16)
    with torch.no_grad():
        after = model(fresh())
    # Outputs are not bit-reproducible run to run (GPU scatter ops), so the exact
    # no-op check is on the weights: effective weight must equal the frozen original.
    for m in model.modules():
        if isinstance(m, torch.nn.MultiheadAttention):
            assert torch.equal(m.in_proj_weight, m.parametrizations.in_proj_weight.original), \
                "LoRA is not a no-op at init"
    drift = (before - after).abs().max().item()

    model(fresh()).mean().backward()
    grads = [n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert grads and all("lora_" in n for n in grads), "gradients missing or leaking"

    print(f"weights: {'released checkpoint' if ckpt else 'RANDOM (no checkpoint given)'}")
    print(f"output {tuple(before.shape)}, trainable {trainable}/{total} ({100 * trainable / total:.3f}%)")
    print(f"LoRA no-op at init: yes (weights bit-equal; output drift {drift:.1e} is run-to-run noise)")
    print(f"LoRA tensors receiving gradient: {len(grads)}")


if __name__ == "__main__":
    main()

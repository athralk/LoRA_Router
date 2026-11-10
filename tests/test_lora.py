"""Smoke test on a stand-in with CircuitFormer's attention shapes
(four nn.MultiheadAttention blocks, dim 16/32/64/128, heads dim//4, 8 latents).
Run: python tests/test_lora.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lora_router.lora import apply_lora, lora_state_dict, merge_lora  # noqa: E402


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 8)
        self.mhsa = nn.MultiheadAttention(dim, num_heads=dim // 4, batch_first=True)
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        return self.head(self.mhsa(x, x, x)[0])


def main():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blocks = nn.ModuleList(Block(d) for d in (16, 32, 64, 128)).to(dev)
    xs = [torch.randn(2, 8, d, device=dev) for d in (16, 32, 64, 128)]
    run = lambda: [b(x) for b, x in zip(blocks, xs)]

    before = run()
    trainable, total = apply_lora(blocks, rank=8, alpha=16,
                                  linear_targets=("mhsa.out_proj", "score"))
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(before, run())), "not a no-op at init"

    names = [n for n, p in blocks.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names), "non-LoRA param is trainable"

    opt = torch.optim.Adam([p for p in blocks.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(5):
        opt.zero_grad()
        sum(o.pow(2).mean() for o in run()).backward()
        opt.step()
    tuned = run()
    assert not torch.allclose(before[0], tuned[0], atol=1e-4), "LoRA did not train"

    n_adapter = len(lora_state_dict(blocks))
    merge_lora(blocks)
    blocks.eval()
    with torch.no_grad():
        merged = run()
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(tuned, merged)), "merge changed output"

    print(f"OK on {dev}: trainable {trainable}/{total} ({100 * trainable / total:.1f}%), "
          f"{n_adapter} adapter tensors, merge exact")


if __name__ == "__main__":
    main()

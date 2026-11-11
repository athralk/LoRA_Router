"""LoRA for CircuitFormer.

CircuitFormer's attention is nn.MultiheadAttention with a packed q/k/v weight
(in_proj_weight, shape [3d, d]), so there are no separate q/v Linears to wrap.
We add the low-rank update as a weight parametrization instead: the module's
forward is untouched and the adapter can be merged back for zero inference cost.
"""
import math

import torch
import torch.nn as nn
from torch.nn.utils import parametrize


class QVLoRA(nn.Module):
    """Low-rank update on the q and v slices of a packed [3d, d] qkv weight."""

    def __init__(self, dim: int, rank: int, alpha: float):
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(2, rank, dim))
        self.lora_B = nn.Parameter(torch.zeros(2, dim, rank))  # zero => no-op at init
        for a in self.lora_A:
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, weight):
        delta = (torch.bmm(self.lora_B, self.lora_A) * self.scale).to(weight.dtype)  # [2, d, d]
        return weight + torch.cat([delta[0], torch.zeros_like(delta[0]), delta[1]], dim=0)


class LinearLoRA(nn.Module):
    """Low-rank update on an [out, in] Linear weight."""

    def __init__(self, out_dim: int, in_dim: int, rank: int, alpha: float):
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, weight):
        return weight + (self.lora_B @ self.lora_A * self.scale).to(weight.dtype)


def apply_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0,
               linear_targets: tuple = (), train_also: tuple = (), adapter_dtype=None):
    """Freeze `model` and attach LoRA.

    Every nn.MultiheadAttention gets q/v LoRA. `linear_targets` are name
    suffixes of extra nn.Linear layers to adapt (e.g. "mhsa.out_proj", "score",
    or "q_proj", "v_proj" for a Hugging Face LLM).
    `train_also` are name substrings of parameters left fully trainable
    (e.g. "decoder.segmentation_head").
    `adapter_dtype` keeps adapters in e.g. float32 on a bf16 model (default: weight dtype).
    Returns (trainable_params, total_params).
    """
    for p in model.parameters():
        p.requires_grad = False

    for name, m in list(model.named_modules()) if rank > 0 else []:
        if isinstance(m, nn.MultiheadAttention) and m.in_proj_weight is not None:
            r = min(rank, m.embed_dim)
            w = m.in_proj_weight
            parametrize.register_parametrization(
                m, "in_proj_weight", QVLoRA(m.embed_dim, r, alpha).to(w.device, adapter_dtype or w.dtype))
        elif isinstance(m, nn.Linear) and any(name.endswith(t) for t in linear_targets):
            r = min(rank, m.in_features, m.out_features)
            parametrize.register_parametrization(
                m, "weight", LinearLoRA(m.out_features, m.in_features, r, alpha)
                .to(m.weight.device, adapter_dtype or m.weight.dtype))

    for name, p in model.named_parameters():
        if "lora_" in name or any(t in name for t in train_also):
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def lora_state_dict(model: nn.Module):
    """Only the trainable tensors: this is the adapter file to save per node."""
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def merge_lora(model: nn.Module):
    """Fold adapters into the base weights and remove the parametrizations."""
    for m in model.modules():
        if parametrize.is_parametrized(m):
            for pname in list(m.parametrizations.keys()):
                parametrize.remove_parametrizations(m, pname, leave_parametrized=True)
    return model

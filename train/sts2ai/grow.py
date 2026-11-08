"""Function-preserving growth of a checkpoint to a bigger preset (more layers and/or a wider feed-forward).

    python -m sts2ai.grow --ckpt runs/all_chars/latest.pt --preset grown --out runs/all_chars_grown/init.pt [--obs obs.bin]

The grown network computes exactly what the old one does, so training continues from the same policy with
room to use the extra capacity:
  * existing blocks keep their weights; extra feed-forward units get fresh input weights and zero output weights;
  * new blocks are appended as identities: pre-LN residual blocks whose output projections (proj, ff2) are zero.
Both the weights and their EMA are grown; the optimizer starts fresh. With --obs (from `Sts2Sim.dll dumpobs`)
the old and grown outputs are compared and the script fails loudly if they differ.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict

import torch

from .model import PRESETS, ModelConfig, Sts2Net


def grow_state(old: dict, old_cfg: ModelConfig, new: dict, new_cfg: ModelConfig) -> dict:
    """Map an old state dict onto a freshly initialised grown one (`new`, modified and returned)."""
    f0, f1 = old_cfg.d_ff, new_cfg.d_ff
    for name, t in new.items():
        if name.startswith("blocks."):
            i = int(name.split(".")[1])
            if i >= old_cfg.n_layers:
                if name.split(".")[2] in ("proj", "ff2"):
                    t.zero_()  # identity block: no contribution to the residual stream
                continue
            o = old[name]
            if name.endswith("ff1.weight"):
                t[:f0] = o
            elif name.endswith("ff1.bias"):
                t[:f0] = o
            elif name.endswith("ff2.weight"):
                t[:, :f0] = o
                t[:, f0:] = 0  # extra units add nothing until trained
            else:
                t.copy_(o)
            continue
        if name not in old or old[name].shape != t.shape:
            raise SystemExit(f"cannot grow {name}: {tuple(old[name].shape) if name in old else 'missing'} -> {tuple(t.shape)}")
        t.copy_(old[name])
    return new


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--obs", default="")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    old_cfg = ModelConfig(**ck["mcfg"])
    new_cfg = ModelConfig(**{**asdict(PRESETS[a.preset]), "vocab_size": old_cfg.vocab_size})
    for k in ("d_model", "n_heads", "n_memory", "recurrent"):
        if getattr(old_cfg, k) != getattr(new_cfg, k):
            raise SystemExit(f"{k} differs ({getattr(old_cfg, k)} -> {getattr(new_cfg, k)}): not function-preserving")
    if new_cfg.n_layers < old_cfg.n_layers or new_cfg.d_ff < old_cfg.d_ff:
        raise SystemExit("the new preset must not be smaller")

    torch.manual_seed(0)
    template = Sts2Net(new_cfg).state_dict()
    fresh = {k: v.clone() for k, v in template.items()}
    model_sd = grow_state(ck["model"], old_cfg, {k: v.clone() for k, v in fresh.items()}, new_cfg)
    ema_sd = grow_state(ck["ema"], old_cfg, {k: v.clone() for k, v in fresh.items()}, new_cfg) if ck.get("ema") else None

    if a.obs:
        from .export import read_obs
        old_m, new_m = Sts2Net(old_cfg).eval(), Sts2Net(new_cfg).eval()
        old_m.load_state_dict(ck["model"])
        new_m.load_state_dict(model_sd)
        worst = 0.0
        for tc, tn, ac, ar, an in read_obs(a.obs):
            t = lambda x, dt: torch.from_numpy(x).unsqueeze(0).to(dt)
            args = (t(tc, torch.long), t(tn, torch.float32), t(ac, torch.long), t(ar, torch.long), t(an, torch.float32))
            o1, o2 = old_m(*args), new_m(*args)
            for k in ("logits", "v_return", "v_win", "v_floor", "v_hp"):
                worst = max(worst, (o1[k] - o2[k]).abs().max().item())
        print(f"max |output difference| old vs grown: {worst:.2e}")
        if worst > 1e-4:
            raise SystemExit("grown network is not equivalent")

    cfg = dict(ck.get("cfg", {}))
    cfg["preset"] = a.preset
    torch.save({"model": model_sd, "mcfg": asdict(new_cfg), "cfg": cfg, "iter": ck.get("iter", 0),
                "total_steps": ck.get("total_steps", 0), "ema": ema_sd}, a.out)
    n_old = sum(v.numel() for v in ck["model"].values())
    n_new = sum(v.numel() for v in model_sd.values())
    print(f"grew {n_old / 1e6:.2f}M -> {n_new / 1e6:.2f}M params ({old_cfg.n_layers}x{old_cfg.d_ff} -> "
          f"{new_cfg.n_layers}x{new_cfg.d_ff}); wrote {a.out}")


if __name__ == "__main__":
    main()

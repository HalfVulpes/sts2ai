"""Convert a dense checkpoint into a character-routed mixture of experts (function-preserving).

    python -m sts2ai.moe --ckpt runs/weak3/best_1142M.pt --vocab ../gamebin/v0.107.1/vocab.json --out runs/moe/init.pt [--obs obs.bin]

The top `--layers` blocks get one feed-forward per character (model.ModelConfig.moe_layers), each a copy of the
dense one, so the new network computes exactly what the old one does; the experts then diverge with each
character's own gradients while attention, embeddings and heads stay shared. Weights, EMA and the AdamW moments are
converted (each expert starts with the dense layer's moments, so training continues smoothly). With --obs (from `Sts2Sim.dll dumpobs`) old and new outputs are compared, also per
character through export's dense slice, and the script fails loudly if they differ.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from .model import CHARACTER_KEYS, ModelConfig, Sts2Net, dense_for_character, expert_ids_from_vocab


def to_moe(dense: dict, cfg: ModelConfig) -> dict:
    """Dense state dict -> MoE state dict (expert feed-forwards = copies of the dense one)."""
    E, out = len(cfg.expert_ids), {}
    for k, v in dense.items():
        name = moe_name(k, cfg)
        out[name] = v.unsqueeze(0).repeat(E, *([1] * v.dim())).clone() if name != k else v.clone()
    return out


def moe_name(k: str, cfg: ModelConfig) -> str:
    parts = k.split(".")
    if parts[0] == "blocks" and int(parts[1]) >= cfg.n_layers - cfg.moe_layers and parts[2] in ("ff1", "ff2"):
        return f"blocks.{parts[1]}.ffe{parts[2][2]}_{'w' if parts[3] == 'weight' else 'b'}"
    return k


def opt_to_moe(opt: dict, old_cfg: ModelConfig, cfg: ModelConfig) -> dict:
    """AdamW state (indexed by parameter order) -> MoE: expert tensors get E copies of the dense moments.
    The orders differ (a block's own expert parameters come before its submodules'), so map by name."""
    dense_names = [n for n, _ in Sts2Net(old_cfg).named_parameters()]
    new_index = {n: j for j, (n, _) in enumerate(Sts2Net(cfg).named_parameters())}
    if sorted(moe_name(n, cfg) for n in dense_names) != sorted(new_index):
        raise SystemExit("dense and MoE parameters do not correspond: cannot map optimizer state")
    (group,) = opt["param_groups"]
    E, state = len(cfg.expert_ids), {}
    for i, d in enumerate(dense_names):
        s = opt["state"].get(i)
        if s is None:
            continue
        m = moe_name(d, cfg)
        if m != d:
            s = {k: v.unsqueeze(0).repeat(E, *([1] * v.dim())).clone() if torch.is_tensor(v) and v.dim() > 0 else v
                 for k, v in s.items()}
        state[new_index[m]] = s
    return {"state": state, "param_groups": [{**group, "params": list(range(len(new_index)))}]}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True, help="gamebin vocab.json (character ids)")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--obs", default="")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    old_cfg = ModelConfig(**ck["mcfg"])
    if old_cfg.moe_layers:
        raise SystemExit("checkpoint is already MoE")
    ids = expert_ids_from_vocab(json.load(open(a.vocab))["ids"])
    cfg = ModelConfig(**{**asdict(old_cfg), "moe_layers": a.layers, "expert_ids": ids})
    model_sd = to_moe(ck["model"], cfg)
    ema_sd = to_moe(ck["ema"], cfg) if ck.get("ema") else None
    new_m = Sts2Net(cfg).eval()
    new_m.load_state_dict(model_sd)  # strict: every tensor mapped

    if a.obs:
        from .export import read_obs
        old_m = Sts2Net(old_cfg).eval()
        old_m.load_state_dict(ck["model"])
        slices = [dense_for_character(new_m, c) for c in range(len(CHARACTER_KEYS))]
        worst, n = 0.0, 0
        for tc, tn, ac, ar, an in read_obs(a.obs):
            t = lambda x, dt: torch.from_numpy(x).unsqueeze(0).to(dt)
            args = (t(tc, torch.long), t(tn, torch.float32), t(ac, torch.long), t(ar, torch.long), t(an, torch.float32))
            o1 = old_m(*args)
            for m in [new_m] + slices:
                o2 = m(*args)
                for k in ("logits", "v_return", "v_win", "v_floor", "v_hp"):
                    worst = max(worst, (o1[k] - o2[k]).nan_to_num(posinf=0, neginf=0).abs().max().item())
            n += 1
        print(f"{n} observations: max |output difference| dense vs MoE (and per-character slices): {worst:.2e}")
        if worst > 1e-4:
            raise SystemExit("MoE network is not equivalent")

    run_cfg = dict(ck.get("cfg", {}))
    run_cfg["preset"] = "moe"
    out = {"model": model_sd, "mcfg": asdict(cfg), "cfg": run_cfg, "iter": ck.get("iter", 0),
           "total_steps": ck.get("total_steps", 0), "ema": ema_sd}
    if ck.get("opt"):
        out["opt"] = opt_to_moe(ck["opt"], old_cfg, cfg)
    torch.save(out, a.out)
    n_old = sum(v.numel() for v in ck["model"].values())
    n_new = sum(v.numel() for v in model_sd.values())
    print(f"MoE: {n_old / 1e6:.2f}M -> {n_new / 1e6:.2f}M params ({a.layers} expert layers x {len(ids)} characters "
          f"{dict(zip([k.split('.')[1] for k in CHARACTER_KEYS], ids))}); wrote {a.out}")


if __name__ == "__main__":
    main()

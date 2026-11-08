"""Export a checkpoint for the in-game C# runtime (shared/Inference/PolicyNet.cs).

    python -m sts2ai.export --ckpt runs/x/latest.pt --out model.bin [--obs obs.bin --parity parity.bin]

model.bin: "STS2NET1", int32 json length, json {config, tensors: [{name, shape, offset}]}, float32 data.
With --obs (written by `Sts2Sim.dll dumpobs`), also writes the PyTorch float32 outputs for each
observation to --parity, for `Sts2Sim.dll parity --model model.bin --obs obs.bin --expect parity.bin`.
"""
from __future__ import annotations

import argparse
import json
import struct
from dataclasses import asdict

import numpy as np
import torch

from . import spec
from .model import CHARACTER_KEYS, ModelConfig, Sts2Net, dense_for_character


def load_model(ckpt_path: str, ema: bool = False) -> Sts2Net:
    ck = torch.load(ckpt_path, map_location="cpu")
    model = Sts2Net(ModelConfig(**ck["mcfg"]))
    if ema and ck.get("ema") is None:
        raise SystemExit("checkpoint has no EMA weights")
    model.load_state_dict(ck["ema"] if ema else ck["model"])
    return model.eval()


def write_model(model: Sts2Net, path: str, encode_allies: bool = False):
    tensors, blobs, off = [], [], 0
    for name, t in model.state_dict().items():
        a = t.detach().float().cpu().numpy().astype("<f4")
        tensors.append({"name": name, "shape": list(a.shape), "offset": off})
        blobs.append(a.tobytes())
        off += a.size
    # encode_allies: trained with co-op episodes -> the runtime must encode teammates (StateEncoder.EncodeAllies)
    meta = json.dumps({"config": asdict(model.cfg), "tensors": tensors, "encode_allies": encode_allies}).encode()
    with open(path, "wb") as f:
        f.write(b"STS2NET1")
        f.write(struct.pack("<i", len(meta)))
        f.write(meta)
        for b in blobs:
            f.write(b)


def read_obs(path: str):
    with open(path, "rb") as f:
        assert f.read(8) == b"STS2OBS1"
        (n,) = struct.unpack("<i", f.read(4))
        cases = []
        for _ in range(n):
            nt, na = struct.unpack("<ii", f.read(8))
            rd = lambda dt, k: np.frombuffer(f.read(4 * k), dtype=dt).copy()
            tc = rd("<i4", nt * spec.N_TOK_CAT).reshape(nt, spec.N_TOK_CAT)
            tn = rd("<f4", nt * spec.F_TOK).reshape(nt, spec.F_TOK)
            ac = rd("<i4", na * spec.N_ACT_CAT).reshape(na, spec.N_ACT_CAT)
            ar = rd("<i4", na * 2).reshape(na, 2)
            an = rd("<f4", na * spec.F_ACT).reshape(na, spec.F_ACT)
            cases.append((tc, tn, ac, ar, an))
        return cases


@torch.no_grad()
def write_parity(model: Sts2Net, cases, path: str):
    with open(path, "wb") as f:
        f.write(b"STS2PAR1")
        f.write(struct.pack("<i", len(cases)))
        for tc, tn, ac, ar, an in cases:
            t = lambda a, dt: torch.from_numpy(a).unsqueeze(0).to(dt)
            out = model(t(tc, torch.long), t(tn, torch.float32), t(ac, torch.long), t(ar, torch.long), t(an, torch.float32))
            logits = out["logits"][0].float().numpy().astype("<f4")
            vals = np.array([out["v_return"][0], torch.sigmoid(out["v_win"][0]), out["v_floor"][0], out["v_hp"][0]], "<f4")
            f.write(struct.pack("<i", len(logits)))
            f.write(logits.tobytes())
            f.write(vals.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--obs", default="")
    ap.add_argument("--parity", default="parity.bin")
    ap.add_argument("--ema", action="store_true", help="export the averaged (EMA) weights")
    ap.add_argument("--character", default="", help="MoE checkpoints: export this character's experts as a dense net")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(a.ckpt, a.ema)
    if model.n_experts:
        if not a.character:
            raise SystemExit("MoE checkpoint: pass --character (the C# runtime loads one dense net per character)")
        model = dense_for_character(model, CHARACTER_KEYS.index("CHARACTER." + a.character.upper()))
    elif a.character:
        raise SystemExit("--character is only for MoE checkpoints")
    checkpoint = torch.load(a.ckpt, map_location="cpu")
    config = checkpoint.get("cfg", {})
    # Full training state takes precedence. Parameter-only releases retain the
    # original inference contract separately, without pretending to contain Adam state.
    coop = (config["coop_prob"] > 0 if "coop_prob" in config else
            bool(checkpoint.get("inference_metadata", {}).get("encode_allies", False)))
    del checkpoint
    write_model(model, a.out, encode_allies=coop)
    print(f"wrote {a.out}")
    if a.obs:
        cases = read_obs(a.obs)
        write_parity(model, cases, a.parity)
        print(f"wrote {a.parity} ({len(cases)} cases)")


if __name__ == "__main__":
    main()

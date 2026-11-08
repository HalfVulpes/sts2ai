#!/usr/bin/env python3
"""Train a policy with one of the paper's recipes.

    python3 scripts/train.py a10_release --character ironclad --init base_ironclad.pt --out runs/ironclad_release
    python3 scripts/train.py a10_push --character defect --init defect.pt --out runs/defect_a10
    python3 scripts/train.py a0_failure --character regent --out runs/regent_a0 --seed 41001

Presets (one update = 65,536 optimizer rows; all at lr 1e-4 at A10 or 2e-4 at A0, failure-driven restarts with
uncertainty-weighted prefix choice unless stated):
  a10_release  release continuation: 1,800 updates from the base policy (paper: "Release continuation"; the released
               Ironclad, Defect and Silent policies are the final EMA of these runs)
  a10_longer   longer continuation: the same recipe for 600 updates (paper: "Longer continuation")
  a10_boss2    second-boss practice: 600 updates with the second final boss as its own restart situation, drawn by
               entry HP, scored by the fight and repeated four times (paper: "Second-boss practice")
  a10_push     long A10 training: 3,072 updates (paper: "Long A10 training")
  a0_failure   controlled study, failure-driven restart allocation (A0, 763 updates)
  a0_uniform   controlled study, uniform restart allocation
  a0_fresh     controlled study, fresh starts only

The presets pin every trainer flag of the reported runs, including the end-of-combat HP label of the runs that
predate the won-run label fix (--hp-label-fix 0) and the training seed of the runs (per character, where the paper
reports one run per character; --seed overrides). Pass the protocol's reserved_seeds.json with --reserved-seeds so
that training never draws an evaluation seed, as the reported runs did.

Needs the imported game (tools/import_game.py), the built host (sim/bin/Release/net9.0/Sts2Sim.dll) and a CUDA
PyTorch. Arguments after `--` are passed to the trainer unchanged and override the preset (for example
`-- --n-envs 32 --steps-per-iter 8192 --fixed-update-rows 8192`). The trainer resumes from --init with
`-- --init-mode resume` if you pass a checkpoint it wrote (latest.pt).
"""
import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMON = ["--preset", "base", "--n-envs", "256", "--steps-per-iter", "65536", "--fixed-update-rows", "65536",
          "--minibatch", "4096", "--epochs", "2", "--token-budget", "262144", "--asc-max", "0", "--coop-prob", "0",
          "--gamma", "0.9998", "--lam", "0.99", "--seed-protocol", "canonical", "--train-seed-bits", "48",
          "--fail-nonfinite", "1", "--save-initial", "1", "--practice-sampler", "uncertainty",
          "--mask-unavailable-restarts", "1"]
RESTARTS = {"failure": ["--restart-prob", "0.5", "--focus", "1", "--focus-sampler", "failure"],
            "uniform": ["--restart-prob", "0.5", "--focus", "1", "--focus-sampler", "uniform"],
            "fresh": ["--restart-prob", "0", "--focus", "0", "--focus-sampler", "failure"]}
# The controlled study, the long A10 training and the continuations ran before the won-run HP label fix or kept the
# old label on purpose; second-boss practice used the fixed label.
OLD_HP_LABEL = ["--hp-label-fix", "0"]
BOSS2 = ["--boss2-situation", "1", "--boss2-hp-stratified", "1", "--fight-outcome", "1", "--practice-repeats", "4",
         "--hp-label-fix", "1"]
A10 = dict(ascension=10, lr="0.0001", restarts="failure", needs_init=True)
A0 = dict(ascension=0, lr="0.0002", updates=763, save_every=20, needs_init=False, flags=OLD_HP_LABEL)
PRESETS = {
    "a10_release": dict(A10, updates=1800, save_every=8, flags=OLD_HP_LABEL,
                        seeds={"ironclad": 811090, "defect": 811092, "silent": 811093}),
    "a10_longer": dict(A10, updates=600, save_every=8, flags=OLD_HP_LABEL, seeds={"ironclad": 711090, "regent": 711091}),
    "a10_boss2": dict(A10, updates=600, save_every=8, flags=BOSS2, seeds={"ironclad": 711090, "regent": 711091}),
    "a10_push": dict(A10, updates=3072, save_every=32, flags=OLD_HP_LABEL,
                     seeds={"silent": 711070, "defect": 711071, "necrobinder": 711072}),
    "a0_failure": dict(A0, restarts="failure"),
    "a0_uniform": dict(A0, restarts="uniform"),
    "a0_fresh": dict(A0, restarts="fresh"),
}


def command(preset, character, out, *, init="", updates=None, device="cuda:0", seed=None, reserved_seeds="",
            sim=None, gamebin=None, dotnet=None, extra=()):
    p = PRESETS[preset]
    if p["needs_init"] and not init:
        raise SystemExit(f"preset {preset} continues a trained policy: pass --init <character>.pt")
    if seed is None:
        seed = p.get("seeds", {}).get(character, 1)
    cmd = [sys.executable, "-u", "-m", "sts2ai.ppo",
           "--sim-dll", str(sim or ROOT / "sim/bin/Release/net9.0/Sts2Sim.dll"),
           "--gamebin", str(gamebin or ROOT / "gamebin/v0.107.1"),
           "--dotnet", dotnet or shutil.which("dotnet") or os.path.expanduser("~/.dotnet/dotnet"),
           *COMMON, *RESTARTS[p["restarts"]], *p["flags"],
           "--characters", character, "--ascension", str(p["ascension"]), "--lr", p["lr"],
           "--iters", str(updates or p["updates"]), "--save-every", str(p["save_every"]),
           "--seed", str(seed), "--env-seed", "-1", "--device", device, "--out", str(Path(out).resolve())]
    if init:
        cmd += ["--init", str(Path(init).resolve()), "--init-mode", "weights"]
    if reserved_seeds:  # evaluation seeds that training must never draw (the protocols' reserved_seeds.json)
        cmd += ["--train-seed-limit", "2064377754059776", "--reserved-seeds-file", str(Path(reserved_seeds).resolve())]
    return cmd + list(extra)


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", choices=sorted(PRESETS))
    ap.add_argument("--character", required=True, choices=["ironclad", "silent", "defect", "necrobinder", "regent"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="", help="weights-only .pt (release archive) or a trainer checkpoint")
    ap.add_argument("--updates", type=int, help="override the preset's number of updates")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, help="training seed (default: the reported run's seed, else 1)")
    ap.add_argument("--reserved-seeds", default="", help="JSON list of seed numbers to exclude from training")
    ap.add_argument("--dry-run", action="store_true", help="print the trainer command and exit")
    a = ap.parse_args(argv)
    cmd = command(a.preset, a.character, a.out, init=a.init, updates=a.updates, device=a.device, seed=a.seed,
                  reserved_seeds=a.reserved_seeds, extra=extra)
    print(shlex.join(cmd), flush=True)
    if a.dry_run:
        return 0
    Path(a.out).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "train") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return subprocess.call(cmd, cwd=ROOT / "train", env=env)


if __name__ == "__main__":
    sys.exit(main())

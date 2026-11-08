#!/usr/bin/env python3
"""Greedy zero-search evaluation of a policy on fixed canonical game seeds.

    python3 scripts/evaluate.py --model ironclad.bin --character ironclad --ascension 10 --seeds CHECK00001,CHECK00002
    python3 scripts/evaluate.py --model defect.pt --character defect --ascension 10 --panel panel.json --out eval/defect

Each game is one `Sts2Sim.dll play` run with one forward pass per decision. A `.pt` (weights-only release file or
trainer checkpoint) is exported to `.bin` first, using its EMA weights. Every game counts: engine errors are
reported and counted as non-wins. Results go to <out>/games/<seed>.json and <out>/summary.json; existing game
files are reused, so an interrupted run continues. A panel is a JSON list of seed strings, or an object whose
--panel-key entry is such a list (items may also be objects with a "seed" or "game_seed" field).
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINE = re.compile(r"run 0: won=(True|False) floor=(\d+) act=(\d+) steps=(\d+) err=(.*?) potions_end=")


def wilson(wins, n, z=1.959964):
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load_seeds(seeds, panel, key):
    if seeds:
        return [s.strip() for s in seeds.split(",") if s.strip()]
    data = json.loads(Path(panel).read_text())
    if isinstance(data, dict):
        if not key:
            raise SystemExit(f"panel has keys {sorted(data)}: choose one with --panel-key")
        data = data[key]
    return [x if isinstance(x, str) else str(x.get("seed") or x.get("game_seed")) for x in data]


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="exported policy .bin, or a .pt to export")
    ap.add_argument("--character", required=True)
    ap.add_argument("--ascension", type=int, required=True)
    ap.add_argument("--seeds", help="comma-separated canonical seed strings")
    ap.add_argument("--panel", help="JSON panel file")
    ap.add_argument("--panel-key", default="")
    ap.add_argument("--out", default="eval")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--sim", default=str(ROOT / "sim/bin/Release/net9.0/Sts2Sim.dll"))
    ap.add_argument("--gamebin", default=str(ROOT / "gamebin/v0.107.1"))
    ap.add_argument("--dotnet", default=shutil.which("dotnet") or os.path.expanduser("~/.dotnet/dotnet"))
    a = ap.parse_args()
    if not (a.seeds or a.panel):
        ap.error("give --seeds or --panel")
    seeds = load_seeds(a.seeds, a.panel, a.panel_key)
    out = Path(a.out)
    (out / "games").mkdir(parents=True, exist_ok=True)
    model = Path(a.model)
    if model.suffix == ".pt":
        target = out / (model.stem + ".bin")
        env = dict(os.environ, PYTHONPATH=str(ROOT / "train"))
        subprocess.run([sys.executable, "-m", "sts2ai.export", "--ckpt", str(model.resolve()), "--ema",
                        "--out", str(target.resolve())], cwd=ROOT / "train", env=env, check=True)
        model = target
    env = dict(os.environ, STS2_GAMEBIN=str(Path(a.gamebin).resolve()), DOTNET_PROCESSOR_COUNT="2",
               OMP_NUM_THREADS="1")

    def one(seed):
        path = out / "games" / f"{seed}.json"
        if path.exists():
            return json.loads(path.read_text())
        row = {"seed": seed, "won": False, "error": None}
        p = subprocess.run([a.dotnet, a.sim, "play", "--model", str(model.resolve()), "--character", a.character,
                            "--ascension", str(a.ascension), "--runs", "1", "--game-seed", seed, "--quiet"],
                           capture_output=True, text=True, env=env, timeout=900)
        line = next((l for l in p.stdout.splitlines() if l.startswith("run 0:")), "")
        m = LINE.match(line)
        if p.returncode or not m or m[5] != "-" or f"seed_string={seed} " not in line:
            row["error"] = (line or p.stderr[-300:]).strip()
        else:
            row.update(won=m[1] == "True", floor=int(m[2]), act=int(m[3]), steps=int(m[4]))
        path.write_text(json.dumps(row) + "\n")
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        rows = list(pool.map(one, seeds))
    wins = sum(r["won"] and not r["error"] for r in rows)
    errors = sum(bool(r["error"]) for r in rows)
    lo, hi = wilson(wins, len(rows))
    summary = {"model_sha256": sha(model), "character": a.character, "ascension": a.ascension, "n": len(rows),
               "wins": wins, "errors": errors, "win_rate": wins / len(rows), "wilson95": [lo, hi], "rows": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"{a.character} A{a.ascension}: {wins}/{len(rows)} wins = {100 * wins / len(rows):.2f}% "
          f"(Wilson 95% {100 * lo:.2f}-{100 * hi:.2f}%), {errors} engine errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

# STS2 AI: zero-search reinforcement learning for Slay the Spire 2

Code for the paper **Zero-Search Reinforcement Learning for Slay the Spire 2 with Failure-Driven Restarts**
(Xiangqin Chen and Zhengxuan Xie, University of Waterloo, November 2026; [arXiv:2611.12088](https://arxiv.org/abs/2611.12088)).

The agent chooses every action with one forward pass of a 6.6M-parameter entity transformer, without lookahead.
Training and evaluation run the game's own compiled assembly (Steam v0.107.1) in a headless .NET host, which replays
a run exactly from its seed and action sequence; a failure-driven curriculum restarts training episodes where
complete runs are lost. The same C# inference runs inside the game as a mod.

**No game files are distributed.** You need your own Steam copy of Slay the Spire 2 v0.107.1. The importer
verifies it and stages its assemblies locally; nothing derived from game data is shipped.

## What is where

| Place | Contents |
|---|---|
| This repository | Source code and short docs: headless host, encoder and C# inference, trainer with its restart curricula, mod source and installer, two entry scripts |
| [GitHub release `v1.0`](https://github.com/HalfVulpes/sts2ai/releases/tag/v1.0) | `sts2ai-mod-v1.0.zip`: the playable mod with the five released policies. `ironclad.bin`, `silent.bin`, `defect.bin`, `necrobinder.bin`, `regent.bin`: each released policy's exported network, for quick evaluation. `sts2ai-weights-released-five.tar.gz`: the same five policies as `.bin` and as PyTorch weights, with `MANIFEST.json` (hashes and the records that state them). `SHA256SUMS` for these files |
| [Records bundle](https://drive.google.com/drive/folders/1OP9IzFIYPuyR97eg1XZru6bpZDSh87Oz) | Frozen protocols and seed panels, per-game results behind every table and figure, training-update logs of the released runs, the language-model game logs and the Discussion records, with an index from each paper table to its files |

The released Regent and Necrobinder policies are the base policies of the paper's development history. The
released Ironclad, Defect and Silent policies are the final EMA weights of the release continuation: 1,800 A10
updates from their base policies (preset `a10_release` below).

## Setup

Linux, Python 3.11+, a C compiler, the .NET 9 SDK (`dotnet` on `PATH`), and for training a CUDA build of
PyTorch (`train/requirements.txt`).

```bash
python3 tools/import_game.py --game-dir "/path/to/steamapps/common/Slay the Spire 2"
dotnet build sim/Sts2Sim.csproj -c Release
dotnet sim/bin/Release/net9.0/Sts2Sim.dll vocab --out gamebin/v0.107.1/vocab.json --quiet
```

The importer checks the version and all 11 assembly hashes (`tools/game_v0.107.1_manifest.json`), copies the
assemblies and localization files into `gamebin/v0.107.1/` (ignored by git) and downloads nothing. The vocabulary
is generated from your installation; the released policies expect size 2650.

## Play in the game

Download `sts2ai-mod-v1.0.zip` from the release and run
`tools/install_release_mod.sh sts2ai-mod-v1.0.zip ["/path/to/Slay the Spire 2"]`. It runs the setup steps above if
needed and installs the mod into `<game>/mods/sts2ai/`. F8 toggles autoplay, F9 plays one step, F10 hides the panel.
See [`mod/README.md`](mod/README.md).

## Evaluate a released policy

After the setup above, one `.bin` from the release is enough:

```bash
python3 scripts/evaluate.py --model ironclad.bin --character ironclad \
    --ascension 10 --seeds CHECK00001,CHECK00002,CHECK00003,CHECK00004 --out eval/ironclad
```

Each game is one greedy run of the host's `play` command, which you can also call directly:
`dotnet sim/bin/Release/net9.0/Sts2Sim.dll play --model ironclad.bin --character ironclad --ascension 10 --runs 1
--game-seed CHECK00001 --quiet`. The script accepts a `.pt` too (exported first), takes a panel file from the
records bundle with `--panel <file> --panel-key <name>`, writes one JSON per game and prints the wins with a
Wilson 95% interval. Every game counts; engine errors are reported.

## Train

`scripts/train.py` runs the trainer with the flags of a reported run and prints the exact command
(`python -m sts2ai.ppo ...`, run from `train/`):

| Preset | Reproduces | Starts from | Updates (65,536 rows each) |
|---|---|---|---|
| `a10_release` | Release continuation: the released Ironclad, Defect and Silent policies | a base policy | 1,800 at A10 |
| `a10_longer` | Longer continuation (Ironclad, Regent) | a base policy | 600 at A10 |
| `a10_boss2` | Second-boss practice (Ironclad, Regent) | a base policy | 600 at A10 |
| `a10_push` | Long A10 training (Silent, Defect, Necrobinder) | a base policy | 3,072 at A10 |
| `a0_failure`, `a0_uniform`, `a0_fresh` | The three arms of the controlled study (Regent, Silent) | a random initialization | 763 at A0 |

The A10 presets use a learning rate of 1e-4 and failure-driven restarts with uncertainty-weighted prefix choice;
`a10_boss2` adds the second-boss situation, entry-HP strata, the per-fight outcome and four repeats of each
practiced fight (`--boss2-situation 1 --boss2-hp-stratified 1 --fight-outcome 1 --practice-repeats 4`). The
controlled study uses 2e-4 at A0. Each preset sets the training seed of the reported run where the paper reports
one run per character, and `--hp-label-fix 0` for the runs that used the original end-of-combat HP label (all but
second-boss practice). `--updates`, `--device`, `--seed` and `--reserved-seeds` override the preset; pass the
protocol's `reserved_seeds.json` from the records bundle with `--reserved-seeds` so that training never draws an
evaluation seed, as the reported runs did. Arguments after `--` go to the trainer unchanged.

```bash
python3 scripts/train.py a10_release --character ironclad --init base_ironclad.pt \
    --reserved-seeds protocols/release_continuation_g4_20261108/reserved_seeds.json --out runs/ironclad_release
python3 scripts/train.py a10_longer --character regent \
    --init sts2ai-weights-released-five/training_inputs/regent.pt \
    --reserved-seeds protocols/route1_ab_20261107/reserved_seeds.json --out runs/regent_longer
python3 scripts/train.py a0_failure --character regent --seed 41001 --out runs/regent_a0
```

The released Regent and Necrobinder policies are their own base policies (`training_inputs/<character>.pt` in the
weights archive). The base Ironclad, Defect and Silent policies are not part of the release; the protocols record
their tensor hashes, which the trainer can check with `-- --expected-initial-sha256 <hash>`. Several reported runs
ran in resumed segments; a preset runs the same trainer flags in one process (a resume restarts the simulator pools,
so a segmented run is not bit-identical to an unbroken one).

Code revisions: the controlled study, the short continuation and long A10 training ran the trainer revision
before route 1, which this trainer reproduces with the route-1 options off and `--hp-label-fix 0` (the presets set
both). The longer continuation, second-boss practice and the release continuation ran route-1 revision `1325776`,
whose trainer, worker and curriculum code this repository contains (`train/sts2ai/curriculum.py`,
`sim/Engine/Worker.cs`).

## Layout

| Path | Contents |
|---|---|
| `sim/` | Headless host: Godot shim, run driver, `play`, `vocab`, `mapcheck`, training worker with the restart pools |
| `shared/` | State/action encoding, decision building and C# inference shared by host and mod |
| `train/` | PPO trainer, environment pool, native collector, restart samplers and curricula, exporter |
| `mod/` | In-game mod source |
| `tools/` | Game importer, mod installer and packer, native stub generator |
| `scripts/` | `train.py`, `evaluate.py` |
| `tests/` | Source tests; they need no game files (`python3 -m pytest -q tests`) |

## Results from code outside this release

The Spire Mentor in-host comparison and the swapped-layer, final-boss search and distillation experiments used an
adapter between Spire Mentor and our host and separate scripts; that code is not part of this release, and their
records are in the records bundle.

## Licence and citation

Original code is licensed under **GPL-2.0-only** ([LICENSE](LICENSE)); see
[third-party notices](THIRD_PARTY_NOTICES.md) and [contribution notes](CONTRIBUTING.md). This repository grants no
licence to Slay the Spire 2 or to third-party material, and the project is not affiliated with or endorsed by
Mega Crit. Please cite the paper; [CITATION.cff](CITATION.cff) describes the software. Contact: Xiangqin Chen,
University of Waterloo, xiangqin.chen@uwaterloo.ca.

```bibtex
@misc{chen2026zerosearch,
  title         = {Zero-Search Reinforcement Learning for Slay the Spire 2 with Failure-Driven Restarts},
  author        = {Chen, Xiangqin and Xie, Zhengxuan},
  year          = {2026},
  eprint        = {2611.12088},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2611.12088},
  url           = {https://arxiv.org/abs/2611.12088}
}
```

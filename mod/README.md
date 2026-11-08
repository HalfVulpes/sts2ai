# STS2 AI mod

An in-game mod that lets the trained policies play Slay the Spire 2 (Steam v0.107.1) in front of you. The mod
decides with the same code as the headless host (`shared/Decisions/DecisionBuilder.cs`, `shared/Encoding`) and
runs the network on the CPU in pure C# (`shared/Inference/PolicyNet.cs`), so it needs no native libraries. A
decision takes about 10-30 ms. The released policies use **no search**: one forward pass per decision, as in
the paper.

## Install

You need your own copy of the game (v0.107.1), python3 (3.11+), unzip and the .NET 9 SDK.

1. Download `sts2ai-mod-v1.0.zip` from the GitHub release and check it against the published SHA256SUMS.
2. In a checkout of this repository, run

   ```
   tools/install_release_mod.sh /path/to/sts2ai-mod-v1.0.zip ["/path/to/Slay the Spire 2"]
   ```

   The game folder defaults to `~/.steam/steam/steamapps/common/Slay the Spire 2`.
3. Start the game, allow mods when it asks, and start or continue a run.

The release zip contains `sts2ai.dll`, `sts2ai.json`, the policies (`model*.bin`), `INSTALL.txt` and `LICENSE`.
It contains no game files. The policies also need the game's entity vocabulary, which is derived from game
data and therefore never distributed. The installer generates it locally:

1. `tools/import_game.py` verifies the version and all 11 assembly hashes of your installation and stages the
   assemblies in `gamebin/v0.107.1/` (ignored by git).
2. `dotnet build sim/Sts2Sim.csproj -c Release` builds the headless host.
3. `dotnet sim/bin/Release/net9.0/Sts2Sim.dll vocab --out gamebin/v0.107.1/vocab.json` writes the vocabulary.
   The installer checks that it has size 2650, the size the released policies were trained with.
4. The installer copies the mod files and the vocabulary, renamed `vocab.dat`, into `<game>/mods/sts2ai/`. The
   game treats every `.json` file in a mod folder as a mod manifest, hence the `.dat` name.

To uninstall, delete `<game>/mods/sts2ai/`.

### Building the mod yourself

```
tools/pack_release_mod.sh <models dir> [out zip] [game dir]
```

builds `sts2ai.dll` in Release against the assemblies of your installation (they are referenced, not copied)
and packs it with the policies in `<models dir>`, named as the mod expects (`model.bin` and
`model_<character>.bin`; the release's `<character>.bin` files are the same policies).

## Use

A panel in the top-right corner, just below the game's top bar, shows the AI's view of the current decision:

- its estimated chance of winning the run (P(win));
- the expected final floor;
- its top choices, with probabilities.

| Button | Key | Action |
|---|---|---|
| **Autoplay: ON/OFF** | F8 | Autoplay on/off. The AI plays one action every 0.8 s; the button turns green while it is on. |
| **Step** | F9 | Play exactly one AI step (works with autoplay off). |
| **Hide** | F10 | Hide the panel. A small **STS2 AI** button stays in the corner; click it to bring the panel back. |

The buttons do the same as the keys, for remote-desktop sessions where function keys do not reach the game.
With autoplay off you play normally, and the panel acts as an advisor.

**Per-character models.** The mod uses `model_<character>.bin` when that file exists (for example
`model_ironclad.bin`); otherwise it uses `model.bin`, which must be present. Runs at ascension 8 or higher use
`model_<character>_hi.bin` when that file exists, otherwise the normal one. The panel's first line names the
model in use. The release zip ships the five released policies as `model_<character>.bin`, with `model.bin` a
copy of the Defect policy, and no `_hi` models.

## Search (off by default)

The paper's policies play without search, and so does the mod as released: the panel shows `Search: n/a`.
An experimental search mode exists for single-player boss and elite fights. It needs a `search.cfg` file in
the mod folder that points at a locally built headless host, which in turn needs your game's assemblies, so
it cannot be shipped. With it, the mod starts `Sts2Sim.dll searchserver`, restores the room from the
room-entry state the game records, replays the AI's decisions in that room, and evaluates the top actions in
forked rollouts at uncertain decisions. It stays off for a room in which you acted by hand, on a timeout, and
in multiplayer; the **Search** button toggles it. Search results are not part of the paper.

`search.cfg` (JSON):

```
{ "dotnet": "/path/to/dotnet",
  "sim": "/path/to/sts2ai/sim/bin/Release/net9.0/Sts2Sim.dll",
  "k": 3, "n": 16, "conf": 0.95, "minp": 0.02, "rooms": "elite", "timeout": 60 }
```

`rooms`: `boss`, `elite` (boss and elite) or `all`. A search uses up to k × n CPU cores for a second or two.

## What it plays

- **Combat:** cards, potions, targets, end turn.
- **Screens:** rewards (gold is claimed automatically), card rewards, map path, events (including Ancients), rest
  sites, shops (buying, card removal), treasure, and card bundles.
- **Card prompts** (discard, exhaust, upgrade, remove, transform, choose-a-card) are answered through the game's
  card-selector hook while autoplay is on.
- **Crystal Sphere:** a simple reveal heuristic.
- **Not played:** the game-over screen, where autoplay turns itself off.
- **Stalemates:** a fight longer than 150 turns pauses autoplay and hands control back to you.

## Strength

The paper reports the win rates of the released policies at Ascension 0 to 10 on held-out seeds (greedy play in
the headless host, no search). The mod plays the same `.bin` files with the same C# code.

## Co-op

The manifest sets `"affects_gameplay": false` because the mod patches no game code; it only reads the screen and
clicks. The game therefore lets a client with the mod join players who don't have it, so the AI can be one seat
in a co-op lobby (it needs its own client and Steam account).
- The policies were trained solo. They do not see teammates, and map/event votes are untested.
- In multiplayer the AI answers card prompts through the game's own (synchronized) selection screens; its
  single-player shortcut (`ICardSelector`) would desync the run.
- If the game reports a desync, note it in the log; it would mean the AI client's state diverged.

## Logs

The game log (`~/.local/share/SlayTheSpire2/logs/godot.log` on Linux) shows every action the AI takes on lines
starting with `[sts2ai]`. If the mod waits more than 5 s on a screen it can't handle, it logs the details
(`waiting on …`). Updating the DLL replaces files atomically: overwriting the loaded `sts2ai.dll` in place
crashes a running game, so restart the game after an update.

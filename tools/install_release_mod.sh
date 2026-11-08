#!/usr/bin/env bash
# Install the STS2 AI mod into your own Slay the Spire 2 (Steam, v0.107.1) installation.
#
#   tools/install_release_mod.sh <sts2ai-mod-v1.0.zip | extracted mod folder> [game dir]
#
# The mod package holds the mod DLL, its manifest and the trained policies, but no game files. The policies
# need the game's entity vocabulary, which is derived from game data and is therefore never distributed: this
# script generates it locally from YOUR installation with the repository's own tools, then copies everything
# into "<game>/mods/sts2ai/". Steps, each skipped when its output already exists:
#   1. verify the installation (version v0.107.1 and all 11 assembly hashes) with tools/import_game.py;
#   2. stage the game assemblies into gamebin/v0.107.1/ of this checkout (local only, ignored by git);
#   3. build the headless host (sim/Sts2Sim.csproj, Release) and run its `vocab` command;
#   4. check that the vocabulary has the size the released policies were trained with (2650);
#   5. copy sts2ai.dll, sts2ai.json, model*.bin and the vocabulary (as vocab.dat) into the mods folder.
# Requirements: Linux, python3 (3.11+), unzip, and the .NET 9 SDK (`dotnet` on PATH or in ~/.dotnet).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=${1:?usage: tools/install_release_mod.sh <mod zip or folder> [game dir]}
GAME=${2:-${GAME:-$HOME/.steam/steam/steamapps/common/Slay the Spire 2}}
EXPECTED_VOCAB_SIZE=2650
GAMEBIN="$ROOT/gamebin/v0.107.1"
VOCAB="$GAMEBIN/vocab.json"
SIM="$ROOT/sim/bin/Release/net9.0/Sts2Sim.dll"

if command -v dotnet >/dev/null 2>&1; then DOTNET=dotnet
elif [ -x "$HOME/.dotnet/dotnet" ]; then DOTNET="$HOME/.dotnet/dotnet"
else echo "error: the .NET 9 SDK is required (https://dot.net); 'dotnet' was not found" >&2; exit 1; fi
command -v python3 >/dev/null || { echo "error: python3 is required" >&2; exit 1; }
[ -d "$GAME" ] || { echo "error: game folder not found: $GAME (pass it as the second argument)" >&2; exit 1; }

# 1-2. verify and stage the user's own game assemblies
python3 "$ROOT/tools/import_game.py" --game-dir "$GAME" --verify-only
if [ ! -d "$GAMEBIN" ]; then
  python3 "$ROOT/tools/import_game.py" --game-dir "$GAME"
fi

# 3. build the host and generate the vocabulary locally
if [ ! -f "$SIM" ]; then
  "$DOTNET" build "$ROOT/sim/Sts2Sim.csproj" -c Release
fi
if [ ! -f "$VOCAB" ]; then
  (cd "$ROOT" && "$DOTNET" "$SIM" vocab --out "$VOCAB" --quiet)
fi

# 4. the policies index this vocabulary; a different size means a different game build or host
size=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['size'])" "$VOCAB")
if [ "$size" != "$EXPECTED_VOCAB_SIZE" ]; then
  echo "error: generated vocabulary has size $size, the released policies need $EXPECTED_VOCAB_SIZE" >&2; exit 1
fi

# 5. unpack the mod package and install it
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
if [ -f "$SRC" ]; then
  unzip -q "$SRC" -d "$STAGE"
  MOD=$(dirname "$(find "$STAGE" -xdev -name sts2ai.dll -print -quit)")
else
  MOD=$SRC
fi
[ -f "$MOD/sts2ai.dll" ] || { echo "error: no sts2ai.dll in $SRC" >&2; exit 1; }
[ -f "$MOD/model.bin" ] || { echo "error: no model.bin in $SRC (the mod needs it as its default policy)" >&2; exit 1; }
MANIFEST="$MOD/sts2ai.json"; [ -f "$MANIFEST" ] || MANIFEST="$ROOT/mod/sts2ai.json"

DEST="$GAME/mods/sts2ai"
mkdir -p "$DEST"
# replace each file atomically (temp file + rename): a running game keeps its loaded copy
put() { cp "$1" "$2.tmp.$$" && mv -f "$2.tmp.$$" "$2"; }
put "$MOD/sts2ai.dll" "$DEST/sts2ai.dll"
put "$MANIFEST" "$DEST/sts2ai.json"
for m in "$MOD"/model*.bin; do put "$m" "$DEST/$(basename "$m")"; done
# vocab.dat, not .json: the game treats every .json file in a mod folder as a mod manifest
put "$VOCAB" "$DEST/vocab.dat"
rm -f "$DEST/vocab.json"
echo "installed into $DEST:"
ls -l "$DEST"
if pgrep -x SlayTheSpire2 >/dev/null 2>&1; then echo "the game is running: restart it to load the mod"; fi
echo "Start the game, allow mods when asked, and start a run: the STS2 AI panel appears in the top-right corner."

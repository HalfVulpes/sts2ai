#!/usr/bin/env bash
# Build the mod from source and pack a release zip that contains no game files and no game-derived data.
#
#   tools/pack_release_mod.sh <models dir> [out zip] [game dir]
#
# <models dir> holds the policies to ship, already named as the mod expects: model.bin (required default),
# model_<character>.bin and optionally model_<character>_hi.bin (used at ascension 8 and above).
# sts2ai.dll is compiled against the game's assemblies in [game dir] (referenced, never copied: the project sets
# Private=false). The vocabulary is NOT packed; tools/install_release_mod.sh generates it on the user's machine.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODELS=${1:?usage: tools/pack_release_mod.sh <models dir> [out zip] [game dir]}
OUT=${2:-$ROOT/dist/sts2ai-mod-v1.0.zip}
GAME=${3:-${GAME:-$HOME/.steam/steam/steamapps/common/Slay the Spire 2}}
if command -v dotnet >/dev/null 2>&1; then DOTNET=dotnet; else DOTNET="$HOME/.dotnet/dotnet"; fi
[ -f "$MODELS/model.bin" ] || { echo "error: $MODELS/model.bin is required" >&2; exit 1; }

"$DOTNET" build "$ROOT/mod/Sts2AiMod/Sts2AiMod.csproj" -c Release -p:GameDir="$GAME" \
  -p:Deterministic=true -p:ContinuousIntegrationBuild=true -p:PathMap="$ROOT=/sts2ai" \
  -p:DebugType=none -p:DebugSymbols=false
DLL="$ROOT/mod/Sts2AiMod/bin/Release/net9.0/sts2ai.dll"

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
D="$STAGE/sts2ai"
mkdir -p "$D"
cp "$DLL" "$D/"
cp "$ROOT/mod/sts2ai.json" "$ROOT/mod/INSTALL.txt" "$ROOT/LICENSE" "$D/"
for m in "$MODELS"/model*.bin; do cp "$m" "$D/"; done
for f in "$D"/*; do
  case $(basename "$f") in
    vocab*|*.pck|sts2.dll|GodotSharp.dll|0Harmony.dll) echo "error: refusing to pack $(basename "$f")" >&2; exit 1;;
  esac
done
(cd "$D" && sha256sum -- * > SHA256SUMS.txt)
mkdir -p "$(dirname "$OUT")"
OUT=$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")
rm -f "$OUT"
# fixed timestamps and no owner names in the archive
find "$STAGE" -xdev -exec touch -h -d '2026-11-09 00:00:00 UTC' {} +
(cd "$STAGE" && TZ=UTC zip -q -X -r -9 "$OUT" sts2ai)
unzip -l "$OUT"
echo "packed $OUT ($(du -h "$OUT" | cut -f1))"

#!/usr/bin/env python3
"""Stage a user's compatible Linux Steam installation. Downloads no game content."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from pck_extract import read_index

ROOT=Path(__file__).resolve().parents[1]
DLLS=["sts2.dll","GodotSharp.dll","0Harmony.dll","SmartFormat.dll","SmartFormat.ZString.dll",
      "MonoMod.Backports.dll","MonoMod.ILHelpers.dll","Sentry.dll","Steamworks.NET.dll",
      "JetBrains.Annotations.dll","System.IO.Hashing.dll"]


def digest(path):
    with path.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game-dir",type=Path,required=True)
    ap.add_argument("--dest",type=Path,default=ROOT/"gamebin/v0.107.1")
    ap.add_argument("--verify-only",action="store_true")
    args=ap.parse_args()
    game=args.game_dir.expanduser().resolve()
    info=json.loads((game/"release_info.json").read_text())
    if info.get("version")!="v0.107.1":
        raise SystemExit(f"Requires v0.107.1; installation reports {info.get('version')!r}")
    data=game/"data_sts2_linuxbsd_x86_64"
    pinned=json.loads((ROOT/"tools/game_v0.107.1_manifest.json").read_text())
    hashes={name:digest(data/name) for name in DLLS}
    if hashes!=pinned["dll_sha256"]:
        changed=[n for n in DLLS if hashes[n]!=pinned["dll_sha256"].get(n)]
        raise SystemExit("Assembly hashes differ from the audited game: "+", ".join(changed))
    if args.verify_only:
        print("Compatible installation: version and all 11 DLL hashes match."); return
    dest=args.dest.expanduser().resolve()
    if dest.exists(): raise SystemExit(f"Destination already exists; refusing to overwrite: {dest}")
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sts2-import-",dir=dest.parent) as temporary:
        staging=Path(temporary)/"gamebin"; staging.mkdir()
        for name in DLLS: shutil.copy2(data/name,staging/name)
        shutil.copy2(game/"release_info.json",staging/"release_info.json")
        if (data/"sts2.deps.json").exists(): shutil.copy2(data/"sts2.deps.json",staging/"sts2.deps.json")
        res=staging/"res"; res.mkdir()
        count=0
        with (game/"SlayTheSpire2.pck").open("rb") as f:
            _,entries=read_index(f)
            for name,offset,size,flags in entries:
                if not name.startswith("res://localization/") or not name.endswith(".json"): continue
                if flags&1: raise RuntimeError("Encrypted localization resource is unsupported")
                target=(res/name[len("res://"):]).resolve()
                if not target.is_relative_to(res): raise RuntimeError("Invalid archive path")
                target.parent.mkdir(parents=True,exist_ok=True)
                f.seek(offset); content=f.read(size)
                if len(content)!=size: raise RuntimeError("Truncated game archive")
                target.write_bytes(content); count+=1
        if not (res/"localization/eng/cards.json").exists():
            raise RuntimeError("English card localization missing")
        (staging/"IMPORT_MANIFEST.json").write_text(json.dumps({"version":"v0.107.1","dll_sha256":hashes,
                                                                 "localization_files":count},indent=2)+"\n")
        staging.rename(dest)
    print(f"Staged {len(DLLS)} assemblies and {count} localization files in {dest}")
    print("Build the host, then run its vocab command to generate vocab.json locally.")


if __name__=="__main__": main()

#!/usr/bin/env python3
"""Minimal Godot 4 .pck reader: list or extract files matching path prefixes/suffixes.

Used to pull the plain-data resources (localization JSON, mp_names.json, ...) that the
headless simulator serves through its virtual res:// filesystem.

usage: pck_extract.py GAME.pck OUT_DIR [--prefix res://localization/] [--suffix .json] [--list]
"""
import argparse
import os
import struct


def read_index(f):
    magic = f.read(4)
    if magic != b"GDPC":
        # self-contained executable: footer holds pck offset
        f.seek(-12, 2)
        off, magic2 = struct.unpack("<q4s", f.read(12))
        assert magic2 == b"GDPC", "not a pck"
        f.seek(f.tell() - 12 - off)
        start = f.tell()
        assert f.read(4) == b"GDPC"
    else:
        start = 0
    fmt_ver, vmaj, vmin, vpatch = struct.unpack("<4I", f.read(16))
    flags = 0
    file_base = 0
    dir_offset = None
    if fmt_ver >= 2:
        flags, file_base = struct.unpack("<IQ", f.read(12))
    if fmt_ver >= 3:
        (dir_offset,) = struct.unpack("<Q", f.read(8))
    if flags & 1:
        raise SystemExit("encrypted pck directory not supported")
    rel_base = bool(flags & 2)
    if dir_offset is not None:
        f.seek(start + dir_offset)
    else:
        f.read(16 * 4)  # reserved
    (count,) = struct.unpack("<I", f.read(4))
    entries = []
    for _ in range(count):
        (plen,) = struct.unpack("<I", f.read(4))
        path = f.read(plen).rstrip(b"\0").decode("utf-8")
        ofs, size = struct.unpack("<QQ", f.read(16))
        f.read(16)  # md5
        eflags = struct.unpack("<I", f.read(4))[0] if fmt_ver >= 2 else 0
        base = file_base + (start if rel_base or fmt_ver >= 3 else 0)
        entries.append((path if path.startswith("res://") else "res://" + path, base + ofs, size, eflags))
    return (vmaj, vmin, vpatch, fmt_ver), entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pck")
    ap.add_argument("out")
    ap.add_argument("--prefix", action="append", default=[])
    ap.add_argument("--suffix", action="append", default=[])
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    with open(a.pck, "rb") as f:
        ver, entries = read_index(f)
        print(f"godot {ver[:3]} pck fmt {ver[3]}: {len(entries)} files")
        n = 0
        for path, ofs, size, eflags in entries:
            if a.prefix and not any(path.startswith(p) for p in a.prefix):
                continue
            if a.suffix and not any(path.endswith(s) for s in a.suffix):
                continue
            n += 1
            if a.list:
                print(f"{size:>10} {path}")
                continue
            if eflags & 1:
                print("skip encrypted", path)
                continue
            dst = os.path.join(a.out, path[len("res://"):])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            f.seek(ofs)
            with open(dst, "wb") as o:
                o.write(f.read(size))
        print(f"{'listed' if a.list else 'extracted'} {n} files")


if __name__ == "__main__":
    main()

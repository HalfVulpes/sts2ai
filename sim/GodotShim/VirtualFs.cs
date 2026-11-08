using System.Diagnostics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using Godot;
using HarmonyLib;
using FileAccess = Godot.FileAccess;

namespace Sts2Sim.GodotShim;

/// <summary>
/// Replaces Godot's FileAccess/DirAccess/OS/Time/ProjectSettings with managed versions.
/// res:// maps to an extracted copy of the game's data files, user:// to a scratch directory.
/// Writes are accepted and discarded (the simulator never needs to persist game saves).
/// </summary>
public static class VirtualFs
{
    public static string ResRoot { get; private set; } = "";
    public static string UserRoot { get; private set; } = "";

    static readonly FieldInfo NativePtrField = typeof(GodotObject).GetField("NativePtr", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)!;
    static readonly ConditionalWeakTable<GodotObject, OpenFile> Files = new();
    static readonly ConditionalWeakTable<GodotObject, DirState> Dirs = new();
    static readonly Stopwatch Clock = Stopwatch.StartNew();
    [ThreadStatic] static Error _lastOpenError;

    sealed class OpenFile
    {
        public byte[] Data = Array.Empty<byte>();
        public long Pos;
        public bool Open = true;
    }

    sealed class DirState
    {
        public string Path = "";
        public List<(string name, bool isDir)> Entries = new();
        public int Cursor = -1;
    }

    public static string Map(string path)
    {
        if (path.StartsWith("res://")) return System.IO.Path.Combine(ResRoot, path[6..]);
        if (path.StartsWith("user://")) return System.IO.Path.Combine(UserRoot, path[7..]);
        return path;
    }

    static T NewObj<T>() where T : GodotObject
    {
        var o = (T)RuntimeHelpers.GetUninitializedObject(typeof(T));
        NativePtrField.SetValue(o, NativeStubs.NewFakeObject());
        GC.SuppressFinalize(o);
        return o;
    }

    public static void Apply(Harmony h, string resRoot, string userRoot)
    {
        ResRoot = resRoot;
        UserRoot = userRoot;
        Directory.CreateDirectory(userRoot);
        void P(Type t, string name, string prefix, Type[]? args = null)
        {
            var m = (args == null ? t.GetMethod(name, BindingFlags.Public | BindingFlags.Static | BindingFlags.Instance) : t.GetMethod(name, args))
                    ?? throw new MissingMethodException(t.Name, name);
            h.Patch(m, prefix: new HarmonyMethod(typeof(VirtualFs), prefix));
        }
        var fa = typeof(FileAccess);
        P(fa, nameof(FileAccess.Open), nameof(FaOpen));
        P(fa, nameof(FileAccess.GetOpenError), nameof(FaGetOpenError));
        P(fa, nameof(FileAccess.FileExists), nameof(FaFileExists));
        P(fa, nameof(FileAccess.GetModifiedTime), nameof(FaGetModifiedTime));
        P(fa, nameof(FileAccess.GetSize), nameof(FaGetSize));
        P(fa, nameof(FileAccess.IsOpen), nameof(FaIsOpen));
        P(fa, nameof(FileAccess.Seek), nameof(FaSeek));
        P(fa, nameof(FileAccess.GetPosition), nameof(FaGetPosition));
        P(fa, nameof(FileAccess.GetLength), nameof(FaGetLength));
        P(fa, nameof(FileAccess.EofReached), nameof(FaEof));
        P(fa, nameof(FileAccess.GetBuffer), nameof(FaGetBuffer));
        P(fa, nameof(FileAccess.GetLine), nameof(FaGetLine));
        P(fa, nameof(FileAccess.GetAsText), nameof(FaGetAsText));
        P(fa, nameof(FileAccess.GetError), nameof(FaGetError));
        P(fa, nameof(FileAccess.StoreBuffer), nameof(FaStore), new[] { typeof(byte[]) });
        P(fa, nameof(FileAccess.StoreBuffer), nameof(FaStore), new[] { typeof(ReadOnlySpan<byte>) });
        P(fa, nameof(FileAccess.StoreLine), nameof(FaStore));
        P(fa, nameof(FileAccess.Flush), nameof(Skip));
        P(fa, nameof(FileAccess.Close), nameof(FaClose));

        var da = typeof(DirAccess);
        P(da, nameof(DirAccess.Open), nameof(DaOpen));
        P(da, nameof(DirAccess.ListDirBegin), nameof(DaListDirBegin));
        P(da, nameof(DirAccess.GetNext), nameof(DaGetNext));
        P(da, nameof(DirAccess.CurrentIsDir), nameof(DaCurrentIsDir));
        P(da, nameof(DirAccess.GetFiles), nameof(DaGetFiles));
        P(da, nameof(DirAccess.GetDirectories), nameof(DaGetDirectories));
        P(da, nameof(DirAccess.GetFilesAt), nameof(DaGetFilesAt));
        P(da, nameof(DirAccess.GetDirectoriesAt), nameof(DaGetDirectoriesAt));
        P(da, nameof(DirAccess.DirExistsAbsolute), nameof(DaDirExists));
        P(da, nameof(DirAccess.MakeDirAbsolute), nameof(ErrOk));
        P(da, nameof(DirAccess.MakeDirRecursiveAbsolute), nameof(ErrOk));
        P(da, nameof(DirAccess.RenameAbsolute), nameof(ErrOk));
        P(da, nameof(DirAccess.Remove), nameof(ErrOk));
        P(da, nameof(DirAccess.RemoveAbsolute), nameof(ErrOk));

        P(typeof(ProjectSettings), nameof(ProjectSettings.GlobalizePath), nameof(Globalize));
        P(typeof(Time), nameof(Time.GetTicksMsec), nameof(TicksMsec));
        P(typeof(OS), nameof(OS.GetUserDataDir), nameof(UserDataDir));
        P(typeof(OS), nameof(OS.GetDataDir), nameof(UserDataDir));
        P(typeof(OS), nameof(OS.GetExecutablePath), nameof(ExecutablePath));
        P(typeof(OS), nameof(OS.GetName), nameof(OsName));
        P(typeof(OS), nameof(OS.GetLocale), nameof(Locale));
        P(typeof(OS), nameof(OS.GetLocaleLanguage), nameof(Locale));
        P(typeof(OS), nameof(OS.GetProcessorCount), nameof(ProcessorCount));
    }

    static bool Skip() => false;
    static bool ErrOk(ref Error __result) { __result = Error.Ok; return false; }

    // ---------------- FileAccess ----------------
    static bool FaOpen(string path, FileAccess.ModeFlags flags, ref FileAccess? __result)
    {
        var real = Map(path);
        var f = new OpenFile();
        if (flags == FileAccess.ModeFlags.Read)
        {
            if (!File.Exists(real)) { _lastOpenError = Error.FileNotFound; __result = null; return false; }
            f.Data = File.ReadAllBytes(real);
        }
        var obj = NewObj<FileAccess>();
        Files.Add(obj, f);
        _lastOpenError = Error.Ok;
        __result = obj;
        return false;
    }

    static bool FaGetOpenError(ref Error __result) { __result = _lastOpenError; return false; }
    static bool FaFileExists(string path, ref bool __result) { __result = File.Exists(Map(path)); return false; }
    static bool FaGetModifiedTime(string file, ref ulong __result)
    {
        var p = Map(file);
        __result = File.Exists(p) ? (ulong)new DateTimeOffset(File.GetLastWriteTimeUtc(p)).ToUnixTimeSeconds() : 0;
        return false;
    }
    static bool FaGetSize(string file, ref long __result) { var p = Map(file); __result = File.Exists(p) ? new FileInfo(p).Length : -1; return false; }

    static OpenFile F(FileAccess self) => Files.TryGetValue(self, out var f) ? f : new OpenFile { Open = false };
    static bool FaIsOpen(FileAccess __instance, ref bool __result) { __result = F(__instance).Open; return false; }
    static bool FaSeek(FileAccess __instance, ulong position) { F(__instance).Pos = (long)position; return false; }
    static bool FaGetPosition(FileAccess __instance, ref ulong __result) { __result = (ulong)F(__instance).Pos; return false; }
    static bool FaGetLength(FileAccess __instance, ref ulong __result) { __result = (ulong)F(__instance).Data.Length; return false; }
    static bool FaEof(FileAccess __instance, ref bool __result) { var f = F(__instance); __result = f.Pos >= f.Data.Length; return false; }
    static bool FaGetError(ref Error __result) { __result = Error.Ok; return false; }
    static bool FaStore(ref bool __result) { __result = true; return false; }
    static bool FaClose(FileAccess __instance) { F(__instance).Open = false; return false; }

    static bool FaGetBuffer(FileAccess __instance, long length, ref byte[] __result)
    {
        var f = F(__instance);
        long n = Math.Max(0, Math.Min(length, f.Data.Length - f.Pos));
        __result = f.Data.AsSpan((int)f.Pos, (int)n).ToArray();
        f.Pos += n;
        return false;
    }

    static bool FaGetLine(FileAccess __instance, ref string __result)
    {
        var f = F(__instance);
        int start = (int)f.Pos, i = start;
        while (i < f.Data.Length && f.Data[i] != (byte)'\n') i++;
        __result = Encoding.UTF8.GetString(f.Data, start, i - start).TrimEnd('\r');
        f.Pos = Math.Min(f.Data.Length, i + 1);
        return false;
    }

    static bool FaGetAsText(FileAccess __instance, bool skipCr, ref string __result)
    {
        var f = F(__instance);
        var s = Encoding.UTF8.GetString(f.Data);
        if (s.Length > 0 && s[0] == '﻿') s = s[1..];
        __result = skipCr ? s.Replace("\r", "") : s;
        return false;
    }

    // ---------------- DirAccess ----------------
    static bool DaOpen(string path, ref DirAccess? __result)
    {
        var real = Map(path);
        if (!Directory.Exists(real)) { __result = null; return false; }
        var obj = NewObj<DirAccess>();
        var st = new DirState { Path = real };
        foreach (var d in Directory.GetDirectories(real).OrderBy(x => x, StringComparer.Ordinal)) st.Entries.Add((System.IO.Path.GetFileName(d), true));
        foreach (var fl in Directory.GetFiles(real).OrderBy(x => x, StringComparer.Ordinal)) st.Entries.Add((System.IO.Path.GetFileName(fl), false));
        Dirs.Add(obj, st);
        __result = obj;
        return false;
    }

    static DirState D(DirAccess self) => Dirs.TryGetValue(self, out var d) ? d : new DirState();
    static bool DaListDirBegin(DirAccess __instance, ref Error __result) { D(__instance).Cursor = -1; __result = Error.Ok; return false; }
    static bool DaGetNext(DirAccess __instance, ref string __result)
    {
        var d = D(__instance);
        d.Cursor++;
        __result = d.Cursor < d.Entries.Count ? d.Entries[d.Cursor].name : "";
        return false;
    }
    static bool DaCurrentIsDir(DirAccess __instance, ref bool __result)
    {
        var d = D(__instance);
        __result = d.Cursor >= 0 && d.Cursor < d.Entries.Count && d.Entries[d.Cursor].isDir;
        return false;
    }
    static bool DaGetFiles(DirAccess __instance, ref string[] __result) { __result = D(__instance).Entries.Where(e => !e.isDir).Select(e => e.name).ToArray(); return false; }
    static bool DaGetDirectories(DirAccess __instance, ref string[] __result) { __result = D(__instance).Entries.Where(e => e.isDir).Select(e => e.name).ToArray(); return false; }
    static bool DaGetFilesAt(string path, ref string[] __result)
    {
        var p = Map(path);
        __result = Directory.Exists(p) ? Directory.GetFiles(p).Select(System.IO.Path.GetFileName).OrderBy(x => x, StringComparer.Ordinal).ToArray()! : Array.Empty<string>();
        return false;
    }
    static bool DaGetDirectoriesAt(string path, ref string[] __result)
    {
        var p = Map(path);
        __result = Directory.Exists(p) ? Directory.GetDirectories(p).Select(System.IO.Path.GetFileName).OrderBy(x => x, StringComparer.Ordinal).ToArray()! : Array.Empty<string>();
        return false;
    }
    static bool DaDirExists(string path, ref bool __result) { __result = Directory.Exists(Map(path)); return false; }

    // ---------------- misc engine queries ----------------
    static bool Globalize(string path, ref string __result) { __result = Map(path); return false; }
    static bool TicksMsec(ref ulong __result) { __result = (ulong)Clock.ElapsedMilliseconds; return false; }
    static bool UserDataDir(ref string __result) { __result = UserRoot; return false; }
    static bool ExecutablePath(ref string __result) { __result = System.IO.Path.Combine(UserRoot, "SlayTheSpire2"); return false; }
    static bool OsName(ref string __result) { __result = "Linux"; return false; }
    static bool Locale(ref string __result) { __result = "en"; return false; }
    static bool ProcessorCount(ref int __result) { __result = 1; return false; }
}

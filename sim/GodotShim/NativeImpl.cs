#pragma warning disable CS8500
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace Sts2Sim.GodotShim;

/// <summary>
/// Hand-written managed implementations of the few Godot native callbacks whose
/// results the game logic actually depends on: Godot strings (UTF-32 CowData),
/// interned StringName/NodePath handles and printing. Everything else is a
/// no-op stub from NativeStubs.g.cs.
/// </summary>
public static unsafe partial class NativeStubs
{
    /// <summary>When false, Godot print/error output is dropped (training).</summary>
    public static bool EchoPrints = true;

    static nint _table;

    /// <summary>Install the managed callback table into GodotSharp. Call once per process.</summary>
    public static void Install()
    {
        if (_table != 0) return;
        _table = (nint)NativeMemory.AllocZeroed((nuint)(Count * sizeof(nint)));
        Fill((nint*)_table);
        Godot.NativeInterop.NativeFuncs.Initialize(_table, Count * sizeof(nint));
    }

    // ---- godot_string: pointer to UTF-32 data, refcount at -16, size (incl. NUL) at -8 ----

    static nint NewGodotString(ReadOnlySpan<char> s)
    {
        if (s.Length == 0) return 0;
        int n = Encoding.UTF32.GetByteCount(s) / 4;
        byte* block = (byte*)NativeMemory.Alloc((nuint)(16 + (n + 1) * 4));
        *(long*)block = 1;
        *(long*)(block + 8) = n + 1;
        uint* data = (uint*)(block + 16);
        fixed (char* src = s) Encoding.UTF32.GetBytes(src, s.Length, (byte*)data, n * 4);
        data[n] = 0;
        return (nint)data;
    }

    internal static string ReadGodotString(nint str)
    {
        if (str == 0) return string.Empty;
        long size = *(long*)(str - 8);
        if (size <= 1) return string.Empty;
        return Encoding.UTF32.GetString((byte*)str, (int)(size - 1) * 4);
    }

    static string ReadStrPtr(nint pstr) => pstr == 0 ? string.Empty : ReadGodotString(*(nint*)pstr);

    [UnmanagedCallersOnly]
    static void Impl_godotsharp_string_new_with_utf16_chars(nint r_dest, nint p_contents)
    {
        var s = p_contents == 0 ? ReadOnlySpan<char>.Empty : MemoryMarshal.CreateReadOnlySpanFromNullTerminated((char*)p_contents);
        *(nint*)r_dest = NewGodotString(s);
    }

    [UnmanagedCallersOnly]
    static void Impl_godotsharp_string_new_copy(nint r_dest, nint p_src)
    {
        nint data = *(nint*)p_src;
        if (data != 0) Interlocked.Increment(ref *(long*)(data - 16));
        *(nint*)r_dest = data;
    }

    [UnmanagedCallersOnly]
    static void Impl_godotsharp_string_destroy(nint p_self)
    {
        nint data = *(nint*)p_self;
        if (data == 0) return;
        if (Interlocked.Decrement(ref *(long*)(data - 16)) == 0)
            NativeMemory.Free((void*)(data - 16));
        *(nint*)p_self = 0;
    }

    // ---- StringName / NodePath: interned, never freed; equal text => equal handle (Godot semantics) ----

    static readonly ConcurrentDictionary<string, nint> _internIds = new();
    static readonly ConcurrentDictionary<nint, string> _internText = new();
    static long _nextIntern = 0x1000;

    static nint Intern(string s)
    {
        if (s.Length == 0) return 0;
        return _internIds.GetOrAdd(s, key =>
        {
            nint id = (nint)Interlocked.Add(ref _nextIntern, 16);
            _internText[id] = key;
            return id;
        });
    }

    internal static string InternText(nint id) => id == 0 ? string.Empty : _internText.GetValueOrDefault(id, string.Empty);

    [UnmanagedCallersOnly]
    static void Impl_godotsharp_string_name_new_from_string(nint r_dest, nint p_name) => *(nint*)r_dest = Intern(ReadStrPtr(p_name));

    [UnmanagedCallersOnly]
    static void Impl_godotsharp_string_name_new_copy(nint r_dest, nint p_src) => *(nint*)r_dest = *(nint*)p_src;

    [UnmanagedCallersOnly]
    static void Impl_godotsharp_string_name_destroy(nint p_self) => *(nint*)p_self = 0;

    [UnmanagedCallersOnly]
    static void Impl_godotsharp_string_name_as_string(nint r_dest, nint p_name) => *(nint*)r_dest = NewGodotString(InternText(*(nint*)p_name));

    [UnmanagedCallersOnly]
    static void Impl_godotsharp_node_path_new_from_string(nint r_dest, nint p_name) => *(nint*)r_dest = Intern(ReadStrPtr(p_name));

    [UnmanagedCallersOnly]
    static void Impl_godotsharp_node_path_new_copy(nint r_dest, nint p_src) => *(nint*)r_dest = *(nint*)p_src;

    [UnmanagedCallersOnly]
    static void Impl_godotsharp_node_path_destroy(nint p_self) => *(nint*)p_self = 0;

    [UnmanagedCallersOnly]
    static void Impl_godotsharp_node_path_as_string(nint r_dest, nint p_np) => *(nint*)r_dest = NewGodotString(InternText(*(nint*)p_np));

    // ---- string helpers used by managed code paths ----

    [UnmanagedCallersOnly]
    static void Impl_godotsharp_string_simplify_path(nint p_self, nint r_dest) => *(nint*)r_dest = NewGodotString(ReadStrPtr(p_self).Replace('\\', '/').Replace("//", "/"));

    [UnmanagedCallersOnly]
    static void Impl_godotsharp_string_to_camel_case(nint p_self, nint r_dest) => *(nint*)r_dest = NewGodotString(CaseConv(ReadStrPtr(p_self), 0));

    [UnmanagedCallersOnly]
    static void Impl_godotsharp_string_to_pascal_case(nint p_self, nint r_dest) => *(nint*)r_dest = NewGodotString(CaseConv(ReadStrPtr(p_self), 1));

    [UnmanagedCallersOnly]
    static void Impl_godotsharp_string_to_snake_case(nint p_self, nint r_dest) => *(nint*)r_dest = NewGodotString(CaseConv(ReadStrPtr(p_self), 2));

    static string CaseConv(string s, int mode)
    {
        var words = new List<string>();
        var cur = new StringBuilder();
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            bool boundary = c is '_' or ' ' or '-';
            if (boundary) { if (cur.Length > 0) { words.Add(cur.ToString()); cur.Clear(); } continue; }
            if (char.IsUpper(c) && cur.Length > 0 && !char.IsUpper(s[i - 1])) { words.Add(cur.ToString()); cur.Clear(); }
            cur.Append(c);
        }
        if (cur.Length > 0) words.Add(cur.ToString());
        if (mode == 2) return string.Join("_", words.Select(w => w.ToLowerInvariant()));
        var sb = new StringBuilder();
        for (int i = 0; i < words.Count; i++)
        {
            string w = words[i].ToLowerInvariant();
            if (i == 0 && mode == 0) sb.Append(w);
            else sb.Append(char.ToUpperInvariant(w[0])).Append(w.AsSpan(1));
        }
        return sb.ToString();
    }

    // ---- printing ----

    static void Echo(string prefix, string s)
    {
        if (EchoPrints) Console.Error.WriteLine(prefix + s);
    }

    [UnmanagedCallersOnly] static void Impl_godotsharp_print(nint p) => Echo("", ReadStrPtr(p));
    [UnmanagedCallersOnly] static void Impl_godotsharp_print_rich(nint p) => Echo("", ReadStrPtr(p));
    [UnmanagedCallersOnly] static void Impl_godotsharp_printt(nint p) => Echo("", ReadStrPtr(p));
    [UnmanagedCallersOnly] static void Impl_godotsharp_prints(nint p) => Echo("", ReadStrPtr(p));
    [UnmanagedCallersOnly] static void Impl_godotsharp_printraw(nint p) => Echo("", ReadStrPtr(p));
    [UnmanagedCallersOnly] static void Impl_godotsharp_printerr(nint p) => Echo("[err] ", ReadStrPtr(p));
    [UnmanagedCallersOnly] static void Impl_godotsharp_pusherror(nint p) => Echo("[push_error] ", ReadStrPtr(p));
    [UnmanagedCallersOnly] static void Impl_godotsharp_pushwarning(nint p) => Echo("[push_warning] ", ReadStrPtr(p));

    [UnmanagedCallersOnly]
    static void Impl_godotsharp_err_print_error(nint func, nint file, int line, nint err, nint descr, byte editorNotify, int type)
        => Echo("[godot_error] ", ReadStrPtr(err) + " " + ReadStrPtr(descr) + " @" + ReadStrPtr(func));

    [UnmanagedCallersOnly]
    static void Impl_godotsharp_internal_script_debugger_send_error(nint func, nint file, int line, nint err, nint descr, int type, nint stack)
        => Echo("[godot_error] ", ReadStrPtr(err) + " " + ReadStrPtr(descr));

    [UnmanagedCallersOnly]
    static byte Impl_godotsharp_dotnet_module_is_initialized() => 1;

    /// <summary>Names of stubbed engine functions that were called, with counts (diagnostics).</summary>
    public static IEnumerable<(string name, long calls)> UnimplementedCalls()
    {
        for (int i = 0; i < Count; i++)
            if (Calls[i] > 0) yield return (Names[i], Calls[i]);
    }
}

public static unsafe partial class NativeStubs
{
    // Fake, non-null method binds: every engine method call then becomes a no-op ptrcall
    // returning default values instead of throwing NativeMethodBindNotFoundException.
    [UnmanagedCallersOnly]
    static nint Impl_godotsharp_method_bind_get_method(nint cls, nint method) => 1;

    [UnmanagedCallersOnly]
    static nint Impl_godotsharp_method_bind_get_method_with_compatibility(nint cls, nint method, ulong hash) => 1;

    [UnmanagedCallersOnly]
    static nint Impl_godotsharp_engine_get_singleton(nint name) => 0;
}

public static unsafe partial class NativeStubs
{
    static long _nextFakeObject = 0x7f00_0000_0000;

    /// <summary>Unique, never-dereferenced fake native object pointer.</summary>
    public static nint NewFakeObject() => (nint)Interlocked.Add(ref _nextFakeObject, 64);

    [UnmanagedCallersOnly]
    static nint FakeClassCtor(byte initialize) => NewFakeObject();

    [UnmanagedCallersOnly]
    static nint Impl_godotsharp_get_class_constructor(nint className) => (nint)(delegate* unmanaged<byte, nint>)&FakeClassCtor;
}

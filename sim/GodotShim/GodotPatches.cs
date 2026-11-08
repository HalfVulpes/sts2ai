using System.Reflection;
using System.Runtime.CompilerServices;
using HarmonyLib;

namespace Sts2Sim.GodotShim;

/// <summary>
/// Harmony patches on GodotSharp itself so engine singletons (OS, Engine, Time, ...) exist as
/// wrappers around fake native pointers. Their method calls then become no-op ptrcalls that
/// return default values.
/// </summary>
public static class GodotPatches
{
    static readonly Assembly GodotAsm = typeof(Godot.GodotObject).Assembly;
    static readonly FieldInfo NativePtrField = typeof(Godot.GodotObject).GetField("NativePtr", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)!;
    static readonly Dictionary<string, Godot.GodotObject> Singletons = new();

    public static Harmony Harmony { get; } = new("sts2sim.godot");

    public static void Apply()
    {
        var interop = GodotAsm.GetType("Godot.NativeInterop.InteropUtils")!;
        Harmony.Patch(interop.GetMethod("EngineGetSingleton", BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)!,
            prefix: new HarmonyMethod(typeof(GodotPatches), nameof(EngineGetSingletonPrefix)));
        Harmony.Patch(typeof(Godot.Engine).GetMethod(nameof(Godot.Engine.GetMainLoop))!,
            prefix: new HarmonyMethod(typeof(GodotPatches), nameof(GetMainLoopPrefix)));
        Harmony.Patch(typeof(Godot.SceneTree).GetProperty(nameof(Godot.SceneTree.Root))!.GetGetMethod()!,
            prefix: new HarmonyMethod(typeof(GodotPatches), nameof(GetRootPrefix)));
    }

    /// <summary>Wrap a fake native pointer in a managed Godot object of type T without running its constructor.</summary>
    public static T Fake<T>() where T : Godot.GodotObject
    {
        var o = (T)RuntimeHelpers.GetUninitializedObject(typeof(T));
        NativePtrField.SetValue(o, NativeStubs.NewFakeObject());
        GC.SuppressFinalize(o);
        return o;
    }

    static Godot.SceneTree? _tree;
    static Godot.Window? _root;

    static bool GetMainLoopPrefix(ref Godot.MainLoop __result) { __result = _tree ??= Fake<Godot.SceneTree>(); return false; }
    static bool GetRootPrefix(ref Godot.Window __result) { __result = _root ??= Fake<Godot.Window>(); return false; }

    static bool EngineGetSingletonPrefix(string name, ref Godot.GodotObject __result)
    {
        lock (Singletons)
        {
            if (!Singletons.TryGetValue(name, out var obj))
            {
                var type = GodotAsm.GetType("Godot." + name + "Instance") ?? GodotAsm.GetType("Godot." + name)
                           ?? throw new InvalidOperationException("Unknown Godot singleton " + name);
                obj = (Godot.GodotObject)RuntimeHelpers.GetUninitializedObject(type);
                NativePtrField.SetValue(obj, NativeStubs.NewFakeObject());
                Singletons[name] = obj;
            }
            __result = obj;
        }
        return false;
    }
}

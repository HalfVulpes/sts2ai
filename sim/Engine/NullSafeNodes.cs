using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using HarmonyLib;

namespace Sts2Sim.Engine;

/// <summary>
/// Headless, every UI node singleton (NCombatRoom.Instance, NGame.Instance, NDebugAudioManager.Instance, ...)
/// is null. Most game logic guards its presentation code, but some does not and throws a
/// NullReferenceException mid-effect (e.g. SoulNexus death, Kaiser Crab arm moves, several event options).
///
/// This scans the game's non-UI code for member accesses on UI objects (Godot objects, or classes
/// in MegaCrit.Sts2.Core.Nodes.*) and rewrites each one null-safely:
/// <c>recv.M(args)</c> becomes <c>recv == null ? default : recv.M(args)</c> (a completed Task for
/// Task-returning members), and likewise for field loads and Godot extension methods. With a
/// non-null receiver behaviour is unchanged, so this only replaces crashes with "skip the visuals".
/// </summary>
public static class NullSafeNodes
{
    static readonly Dictionary<short, OpCode> OpTable = typeof(OpCodes).GetFields(BindingFlags.Public | BindingFlags.Static)
        .Select(f => (OpCode)f.GetValue(null)!).ToDictionary(o => o.Value);

    public static int Patched { get; private set; }

    static bool IsUiType(Type? t)
    {
        if (t == null || !t.IsClass) return false;
        var ns = t.Namespace ?? "";
        return ns.StartsWith("MegaCrit.Sts2.Core.Nodes") || typeof(Godot.GodotObject).IsAssignableFrom(t);
    }

    static bool InUiCode(Type t)
    {
        while (t.DeclaringType != null) t = t.DeclaringType;
        if (typeof(Godot.GodotObject).IsAssignableFrom(t)) return true;
        var ns = t.Namespace ?? "";
        return ns.StartsWith("MegaCrit.Sts2.Core.Nodes") || ns.StartsWith("MegaCrit.Sts2.Core.AutoSlay")
               || ns.StartsWith("MegaCrit.Sts2.Core.DevConsole") || ns.StartsWith("MegaCrit.Sts2.Core.Timeline");
    }

    /// <summary>Instance member of a UI class, or a game-side extension method taking a UI object first.</summary>
    static bool IsUiCall(MethodBase m)
    {
        if (m is not MethodInfo mi) return false; // ctors (newobj) are not receiver accesses
        if (!mi.IsStatic) return IsUiType(mi.DeclaringType);
        var ps = mi.GetParameters();
        return ps.Length > 0 && mi.IsDefined(typeof(ExtensionAttribute)) && IsUiType(ps[0].ParameterType)
               && !(mi.DeclaringType?.Namespace ?? "").StartsWith("Godot"); // game-side helpers like AddChildSafely
    }

    public static void Apply(Harmony h)
    {
        var asm = typeof(MegaCrit.Sts2.Core.Runs.RunState).Assembly;
        var transpiler = new HarmonyMethod(typeof(NullSafeNodes), nameof(Transpiler));
        foreach (var t in asm.GetTypes())
        {
            if (InUiCode(t) || t.ContainsGenericParameters) continue;
            foreach (var m in t.GetMethods(BindingFlags.DeclaredOnly | BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)
                         .Cast<MethodBase>()
                         .Concat(t.GetConstructors(BindingFlags.DeclaredOnly | BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)))
            {
                if (m.IsAbstract || m.ContainsGenericParameters || !HasUiAccess(m)) continue;
                try
                {
                    h.Patch(m, transpiler: transpiler);
                    Patched++;
                }
                catch (Exception e)
                {
                    Console.Error.WriteLine($"NullSafeNodes: could not patch {t.FullName}.{m.Name}: {e.Message}");
                }
            }
        }
    }

    /// <summary>Fast IL scan (no Harmony) for call/callvirt/ldfld on UI objects.</summary>
    static bool HasUiAccess(MethodBase m)
    {
        byte[]? il;
        try { il = m.GetMethodBody()?.GetILAsByteArray(); } catch { return false; }
        if (il == null) return false;
        var mod = m.Module;
        Type[]? targs = m.DeclaringType?.IsGenericType == true ? m.DeclaringType.GetGenericArguments() : null;
        Type[]? margs = m.IsGenericMethod ? m.GetGenericArguments() : null;
        int i = 0;
        while (i < il.Length)
        {
            short code = il[i] == 0xFE ? (short)(0xFE00 | il[i + 1]) : il[i];
            i += il[i] == 0xFE ? 2 : 1;
            if (!OpTable.TryGetValue(code, out var op)) return false;
            if (op == OpCodes.Call || op == OpCodes.Callvirt)
            {
                try { if (mod.ResolveMethod(BitConverter.ToInt32(il, i), targs, margs) is { } callee && IsUiCall(callee)) return true; } catch { }
            }
            else if (op == OpCodes.Ldfld)
            {
                try { if (IsUiType(mod.ResolveField(BitConverter.ToInt32(il, i), targs, margs)?.DeclaringType)) return true; } catch { }
            }
            i += OperandSize(op, il, i);
        }
        return false;
    }

    static int OperandSize(OpCode op, byte[] il, int i) => op.OperandType switch
    {
        OperandType.InlineNone => 0,
        OperandType.ShortInlineBrTarget or OperandType.ShortInlineI or OperandType.ShortInlineVar => 1,
        OperandType.InlineVar => 2,
        OperandType.InlineI8 or OperandType.InlineR => 8,
        OperandType.InlineSwitch => 4 + 4 * BitConverter.ToInt32(il, i),
        _ => 4,
    };

    static IEnumerable<CodeInstruction> Transpiler(IEnumerable<CodeInstruction> code, ILGenerator gen)
    {
        CodeInstruction? prev = null;
        foreach (var ins in code)
        {
            bool prefixed = prev != null && (prev.opcode == OpCodes.Constrained || prev.opcode == OpCodes.Tailcall || prev.opcode == OpCodes.Readonly);
            prev = ins;
            if (!prefixed && (ins.opcode == OpCodes.Call || ins.opcode == OpCodes.Callvirt) && ins.operand is MethodInfo mi && IsUiCall(mi)
                && mi.GetParameters().All(p => !p.ParameterType.IsByRef))
            {
                foreach (var x in WrapCall(ins, mi, gen)) yield return x;
            }
            else if (ins.opcode == OpCodes.Ldfld && ins.operand is FieldInfo fi && IsUiType(fi.DeclaringType))
            {
                foreach (var x in Guard(ins, Array.Empty<LocalBuilder>(), fi.FieldType, gen)) yield return x;
            }
            else yield return ins;
        }
    }

    static IEnumerable<CodeInstruction> WrapCall(CodeInstruction ins, MethodInfo mi, ILGenerator gen)
    {
        // stack: [receiver, rest...]; stash `rest` in locals so the receiver is on top
        var ps = mi.GetParameters();
        var restTypes = (mi.IsStatic ? ps.Skip(1) : ps).Select(p => p.ParameterType).ToArray();
        var locals = restTypes.Select(t => gen.DeclareLocal(t)).ToArray();
        return Guard(ins, locals, mi.ReturnType, gen);
    }

    /// <summary>Emit: stloc rest; if (recv == null) { pop; push default } else { ldloc rest; ins }</summary>
    static IEnumerable<CodeInstruction> Guard(CodeInstruction ins, LocalBuilder[] rest, Type result, ILGenerator gen)
    {
        var lCall = gen.DefineLabel();
        var lEnd = gen.DefineLabel();
        var first = new List<CodeInstruction>();
        for (int k = rest.Length - 1; k >= 0; k--) first.Add(new CodeInstruction(OpCodes.Stloc, rest[k]));
        first.Add(new CodeInstruction(OpCodes.Dup));
        first[0].MoveLabelsFrom(ins).MoveBlocksFrom(ins);
        foreach (var x in first) yield return x;
        yield return new CodeInstruction(OpCodes.Brtrue, lCall);
        yield return new CodeInstruction(OpCodes.Pop);
        foreach (var x in PushDefault(result, gen)) yield return x;
        yield return new CodeInstruction(OpCodes.Br, lEnd);
        var load = rest.Select(l => new CodeInstruction(OpCodes.Ldloc, l)).ToList();
        if (load.Count > 0) load[0].labels.Add(lCall); else ins.labels.Add(lCall);
        foreach (var x in load) yield return x;
        yield return ins;
        yield return new CodeInstruction(OpCodes.Nop) { labels = { lEnd } };
    }

    static IEnumerable<CodeInstruction> PushDefault(Type t, ILGenerator gen)
    {
        if (t == typeof(void)) yield break;
        if (t == typeof(Task))
        {
            yield return new CodeInstruction(OpCodes.Call, AccessTools.PropertyGetter(typeof(Task), nameof(Task.CompletedTask)));
            yield break;
        }
        if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(Task<>))
        {
            var arg = t.GetGenericArguments()[0];
            foreach (var x in PushDefault(arg, gen)) yield return x;
            yield return new CodeInstruction(OpCodes.Call, typeof(Task).GetMethod(nameof(Task.FromResult))!.MakeGenericMethod(arg));
            yield break;
        }
        if (!t.IsValueType)
        {
            yield return new CodeInstruction(OpCodes.Ldnull);
            yield break;
        }
        var tmp = gen.DeclareLocal(t);
        yield return new CodeInstruction(OpCodes.Ldloca, tmp);
        yield return new CodeInstruction(OpCodes.Initobj, t);
        yield return new CodeInstruction(OpCodes.Ldloc, tmp);
    }
}

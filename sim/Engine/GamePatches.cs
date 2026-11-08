using System.Reflection;
using System.Reflection.Emit;
using HarmonyLib;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Relics;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;

namespace Sts2Sim.Engine;

/// <summary>
/// Patches on the game assembly:
///  * skip disk/profile side effects (saves, replays, progress) — no rule changes;
///  * undo TestMode-only gameplay shortcuts so the simulator matches the real game
///    (Cauldron / Calling Bell fixed test rewards, merchant potion price roll);
///  * route node-only prompts (Crystal Sphere, card bundles) to the driver.
/// </summary>
public static class GamePatches
{
    /// <summary>Set by the driver: handles a card-bundle choice (returns chosen bundle index).</summary>
    public static Func<Player, IReadOnlyList<IReadOnlyList<CardModel>>, int>? BundleChooser;
    /// <summary>Set by the driver: receives a Crystal Sphere minigame to play.</summary>
    public static Action<CrystalSphereMinigame>? SphereCaptured;

    public static void Apply(Harmony h)
    {
        Skip(h, AccessTools.Method(typeof(RunManager), nameof(RunManager.WriteReplay)));
        Skip(h, AccessTools.Method(typeof(SaveManager), nameof(SaveManager.SaveProgressFile)));
        Skip(h, AccessTools.Method("MegaCrit.Sts2.Core.Saves.Managers.ProgressSaveManager:SaveProgress"));
        h.Patch(AccessTools.Method(typeof(SaveManager), nameof(SaveManager.SaveRun)),
            prefix: new HarmonyMethod(typeof(GamePatches), nameof(CompletedTaskPrefix)));
        SaveManager.Instance.InitProfileId(0);
        SaveManager.Instance.InitPrefsDataForTest();

        // TestMode shortcuts that change gameplay -> behave like the real game
        ForceTestMode(h, AccessTools.Method(typeof(Cauldron), "GenerateRewards"));
        ForceTestMode(h, AccessTools.Method(typeof(CallingBell), "GenerateRewards"));
        ForceTestMode(h, AccessTools.Method(typeof(MerchantPotionEntry), nameof(MerchantPotionEntry.CalcCost)));

        h.Patch(AccessTools.Method(typeof(NCrystalSphereScreen), nameof(NCrystalSphereScreen.ShowScreen)),
            prefix: new HarmonyMethod(typeof(GamePatches), nameof(SpherePrefix)));
        h.Patch(AccessTools.Method(typeof(CardSelectCmd), nameof(CardSelectCmd.FromChooseABundleScreen)),
            prefix: new HarmonyMethod(typeof(GamePatches), nameof(BundlePrefix)));

        // real-time waits in game logic -> pump ticks (deterministic, no wall-clock cost)
        foreach (var m in new[] { AccessTools.Method(typeof(MegaCrit.Sts2.Core.GameActions.ActionExecutor), "WaitForUnpause"),
                                  AccessTools.Method(typeof(MegaCrit.Sts2.Core.Models.Events.Amalgamator), "CombineStrikes"),
                                  AccessTools.Method(typeof(MegaCrit.Sts2.Core.Models.Events.Amalgamator), "CombineDefends") })
            h.Patch(AccessTools.AsyncMoveNext(m), transpiler: new HarmonyMethod(typeof(GamePatches), nameof(DelayTranspiler)));

        // CombatState.Creatures/PlayerCreatures/Players build lists through LINQ on every hook iteration
        // (~1/3 of CPU); same results (fresh list, same order) without the LINQ chains
        var cs = typeof(MegaCrit.Sts2.Core.Combat.CombatState);
        h.Patch(AccessTools.PropertyGetter(cs, "Creatures"), prefix: new HarmonyMethod(typeof(GamePatches), nameof(CreaturesPrefix)));
        h.Patch(AccessTools.PropertyGetter(cs, "PlayerCreatures"), prefix: new HarmonyMethod(typeof(GamePatches), nameof(PlayerCreaturesPrefix)));
        h.Patch(AccessTools.PropertyGetter(cs, "Players"), prefix: new HarmonyMethod(typeof(GamePatches), nameof(PlayersPrefix)));

        // exact faster map-generation segment search (was ~60% of CPU)
        FastMapPruning.Apply(h);

        // game logic that touches UI nodes without a TestMode guard: make those accesses null-safe
        NullSafeNodes.Apply(h);
        // co-op: per-player input paths for runs with several players (inactive in single-player runs)
        Coop.Apply(h);
        // Kaiser Crab arms resolve their background node with `?? throw`; with no node, every move would abort
        // before dealing damage. Return null so the (now null-safe) animation calls are skipped.
        foreach (var t in new[] { typeof(MegaCrit.Sts2.Core.Models.Monsters.Crusher), typeof(MegaCrit.Sts2.Core.Models.Monsters.Rocket) })
            h.Patch(AccessTools.PropertyGetter(t, "Background"), prefix: new HarmonyMethod(typeof(GamePatches), nameof(NullResultPrefix)));
    }

    static IEnumerable<CodeInstruction> DelayTranspiler(IEnumerable<CodeInstruction> code)
    {
        var delay = AccessTools.Method(typeof(Task), nameof(Task.Delay), new[] { typeof(int) });
        var tick = AccessTools.Method(typeof(PumpContext), nameof(PumpContext.Delay));
        foreach (var ins in code)
        {
            if (ins.Calls(delay)) ins.operand = tick;
            yield return ins;
        }
    }

    // direct field access (no reflection) for the hot CombatState getters below
    [System.Runtime.CompilerServices.UnsafeAccessor(System.Runtime.CompilerServices.UnsafeAccessorKind.Field, Name = "_allies")]
    static extern ref List<MegaCrit.Sts2.Core.Entities.Creatures.Creature> Allies(MegaCrit.Sts2.Core.Combat.CombatState cs);
    [System.Runtime.CompilerServices.UnsafeAccessor(System.Runtime.CompilerServices.UnsafeAccessorKind.Field, Name = "_enemies")]
    static extern ref List<MegaCrit.Sts2.Core.Entities.Creatures.Creature> Enemies(MegaCrit.Sts2.Core.Combat.CombatState cs);

    static bool CreaturesPrefix(MegaCrit.Sts2.Core.Combat.CombatState __instance, ref IReadOnlyList<MegaCrit.Sts2.Core.Entities.Creatures.Creature> __result)
    {
        var a = Allies(__instance);
        var e = Enemies(__instance);
        var r = new List<MegaCrit.Sts2.Core.Entities.Creatures.Creature>(a.Count + e.Count);
        r.AddRange(a);
        r.AddRange(e);
        __result = r;
        return false;
    }

    static bool PlayerCreaturesPrefix(MegaCrit.Sts2.Core.Combat.CombatState __instance, ref IReadOnlyList<MegaCrit.Sts2.Core.Entities.Creatures.Creature> __result)
    {
        var r = new List<MegaCrit.Sts2.Core.Entities.Creatures.Creature>(2);
        foreach (var c in Allies(__instance)) if (c.IsPlayer) r.Add(c);
        foreach (var c in Enemies(__instance)) if (c.IsPlayer) r.Add(c);
        __result = r;
        return false;
    }

    static bool PlayersPrefix(MegaCrit.Sts2.Core.Combat.CombatState __instance, ref IReadOnlyList<Player> __result)
    {
        var r = new List<Player>(1);
        foreach (var c in Allies(__instance)) if (c.IsPlayer) r.Add(c.Player!);
        foreach (var c in Enemies(__instance)) if (c.IsPlayer) r.Add(c.Player!);
        __result = r;
        return false;
    }

    static bool NullResultPrefix(ref object? __result) { __result = null; return false; }

    static void Skip(Harmony h, MethodBase m) =>
        h.Patch(m, prefix: new HarmonyMethod(typeof(GamePatches), nameof(SkipPrefix)));

    static bool SkipPrefix() => false;

    static bool CompletedTaskPrefix(ref Task __result) { __result = Task.CompletedTask; return false; }

    /// <summary>Inside the given method only, TestMode.IsOn reads false and TestMode.IsOff reads true.</summary>
    static void ForceTestMode(Harmony h, MethodBase m) =>
        h.Patch(m, transpiler: new HarmonyMethod(typeof(GamePatches), nameof(TestModeOffTranspiler)));

    static IEnumerable<CodeInstruction> TestModeOffTranspiler(IEnumerable<CodeInstruction> code)
    {
        var isOn = AccessTools.PropertyGetter(typeof(TestMode), nameof(TestMode.IsOn));
        var isOff = AccessTools.PropertyGetter(typeof(TestMode), nameof(TestMode.IsOff));
        foreach (var ins in code)
        {
            if (ins.Calls(isOn)) yield return new CodeInstruction(OpCodes.Ldc_I4_0).MoveLabelsFrom(ins);
            else if (ins.Calls(isOff)) yield return new CodeInstruction(OpCodes.Ldc_I4_1).MoveLabelsFrom(ins);
            else yield return ins;
        }
    }

    static bool SpherePrefix(CrystalSphereMinigame grid, ref NCrystalSphereScreen? __result)
    {
        __result = null;
        SphereCaptured?.Invoke(grid);
        return false;
    }

    static bool BundlePrefix(Player player, IReadOnlyList<IReadOnlyList<CardModel>> bundles, ref Task<IEnumerable<CardModel>> __result)
    {
        RunManager.Instance.PlayerChoiceSynchronizer.ReserveChoiceId(player);
        int k = BundleChooser?.Invoke(player, bundles) ?? 0;
        if (k < 0 || k >= bundles.Count) k = 0;
        __result = Task.FromResult<IEnumerable<CardModel>>(bundles[k]);
        return false;
    }
}

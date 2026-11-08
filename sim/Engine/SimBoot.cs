using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using Sts2Sim.GodotShim;

namespace Sts2Sim.Engine;

/// <summary>One-time process setup: Godot shim, virtual FS, test mode and the game's own init sequence.</summary>
public static class SimBoot
{
    static bool _done;

    public static void Init(bool quiet, string instanceTag = "")
    {
        if (_done) return;
        _done = true;
        void Trace(string stage) { if (Environment.GetEnvironmentVariable("STS2_BOOT_TRACE") == "1") Console.Error.WriteLine("[boot] " + stage); }
        Trace("native stubs");
        NativeStubs.EchoPrints = !quiet;
        NativeStubs.Install();
        Trace("Godot patches");
        GodotPatches.Apply();
        var gamebin = Environment.GetEnvironmentVariable("STS2_GAMEBIN")
                      ?? Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../gamebin/v0.107.1"));
        Trace("virtual filesystem");
        VirtualFs.Apply(GodotPatches.Harmony, Path.Combine(gamebin, "res"), Path.Combine(Path.GetTempPath(), $"sts2sim-user-{Environment.ProcessId}{instanceTag}"));
        MegaCrit.Sts2.Core.TestSupport.TestMode.TurnOnInternal();
        if (quiet)
        {
            var map = MegaCrit.Sts2.Core.Logging.Logger.logLevelTypeMap;
            foreach (var k in map.Keys.ToList()) map[k] = MegaCrit.Sts2.Core.Logging.LogLevel.Error;
            MegaCrit.Sts2.Core.Logging.Logger.GlobalLogLevel = MegaCrit.Sts2.Core.Logging.LogLevel.Error;
        }
        Trace("game initialization");
        OneTimeInitialization.ExecuteVeryEarly().GetAwaiter().GetResult();
        MegaCrit.Sts2.Core.Localization.LocManager.Initialize();
        ModelDb.Init();
        MegaCrit.Sts2.Core.Multiplayer.Serialization.ModelIdSerializationCache.Init();
        ModelDb.InitIds();
        MegaCrit.Sts2.Core.Multiplayer.Serialization.MessageTypes.Initialize();
        MegaCrit.Sts2.Core.GameActions.Multiplayer.ActionTypes.Initialize();
        ModelDb.Preload();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        Trace("game patches");
        GamePatches.Apply(GodotPatches.Harmony);
        Trace("ready");
        if (!quiet) Console.Error.WriteLine($"[sim] game patches applied ({NullSafeNodes.Patched} methods made UI-null-safe) in {sw.ElapsedMilliseconds} ms");
    }
}

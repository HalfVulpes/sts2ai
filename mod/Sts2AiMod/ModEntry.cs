using Godot;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Modding;

namespace Sts2AiMod;

/// <summary>
/// Mod entry point, called by the game's ModManager after loading sts2ai.dll from mods/sts2ai/.
/// Loads the policy (model.bin) and the training vocabulary (vocab.dat) shipped next to the dll,
/// then runs the controller once per frame on the Godot main thread.
/// </summary>
[ModInitializer(nameof(Init))]
public static class ModEntry
{
    static AiController? _ai;
    static Overlay? _overlay;

    public static void Init()
    {
        try
        {
            string dir = Path.GetDirectoryName(typeof(ModEntry).Assembly.Location)!;
            // vocab.dat: the game treats every .json in a mod folder as a manifest (vocab.json only for old installs)
            var vocab = Path.Combine(dir, "vocab.dat");
            if (!File.Exists(vocab)) vocab = Path.Combine(dir, "vocab.json");
            _ai = new AiController(Path.Combine(dir, "model.bin"), vocab);
            var tree = (SceneTree)Engine.GetMainLoop();
            tree.ProcessFrame += OnFrame;
            Log.Info($"[sts2ai] loaded model from {dir}");
        }
        catch (Exception e)
        {
            Log.Error("[sts2ai] failed to initialize: " + e);
        }
    }

    static void OnFrame()
    {
        try
        {
            var tree = (SceneTree)Engine.GetMainLoop();
            _overlay ??= Overlay.Create(tree.Root, _ai!);
            _ai!.Tick(tree.Root.GetProcessDeltaTime());
            _overlay.Update(_ai);
        }
        catch (Exception e)
        {
            Log.Error("[sts2ai] frame error: " + e);
        }
    }
}

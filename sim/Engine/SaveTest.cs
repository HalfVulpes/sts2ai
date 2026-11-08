using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Multiplayer.Replay;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using Sts2Ai.Shared.Encoding;
using Sts2Ai.Shared.Inference;

namespace Sts2Sim.Engine;

/// <summary>
/// Mirror test (basis of search in the mod): play greedy with the game's CombatReplayWriter on (as in the live
/// game); after --after decisions into the first boss/elite fight, take the room-entry state the writer recorded
/// and the decisions made since; restore a fresh run from that state (StartFromSave), replay the decisions, and
/// compare the observation hash with the original's.
///
///   Sts2Sim.dll savetest --model m.bin --character ironclad --seed 1 --runs 5 [--after 5]
/// </summary>
public static class SaveTest
{
    static readonly System.Reflection.FieldInfo ReplayField = typeof(CombatReplayWriter)
        .GetField("_replay", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
        ?? throw new MissingFieldException("CombatReplayWriter._replay");

    static int Argmax(float[] p) { int b = 0; for (int i = 1; i < p.Length; i++) if (p[i] > p[b]) b = i; return b; }

    /// <summary>The room-entry state recorded by the live game's replay writer (null before the first room).</summary>
    public static CombatReplay? CurrentReplay() => (CombatReplay?)ReplayField.GetValue(RunManager.Instance.CombatReplayWriter);

    public static byte[] Serialize(SerializableRun run)
    {
        var w = new PacketWriter();
        run.Serialize(w);
        return w.Buffer.AsSpan(0, w.BytePosition).ToArray();
    }

    public static SerializableRun Deserialize(byte[] bytes)
    {
        var r = new PacketReader();
        r.Reset(bytes);
        var run = new SerializableRun();
        run.Deserialize(r);
        return run;
    }

    public static void Run(Func<string, string, string> arg, PumpContext pump, Vocab vocab, Action<string> print)
    {
        var net = new PolicyNet(arg("--model", "model.bin"));
        int runs = int.Parse(arg("--runs", "5")), after = int.Parse(arg("--after", "5"));
        int minAct = int.Parse(arg("--min-act", "1")) - 1;   // capture in this act or later
        bool bossOnly = arg("--boss", "0") == "1";
        int character = RunDriver.CharacterIndex(arg("--character", "Ironclad")), asc = int.Parse(arg("--ascension", "0"));
        var rng = new Random(int.Parse(arg("--seed", "1")));
        var driver = new RunDriver(pump, vocab);
        driver.Trace = s => { if (s.StartsWith("ERROR")) print("  " + s.Split('\n')[0]); };
        int match = 0, tested = 0;
        for (int r = 0; r < runs; r++)
        {
            ulong runSeed = (ulong)rng.NextInt64();
            object? lastReplay = null;
            var log = new List<int>();
            byte[]? saved = null;
            List<int>? savedLog = null;
            ulong savedHash = 0;
            string where = "";
            int inFight = 0;
            driver.Policy = (d, enc) =>
            {
                var rep = CurrentReplay();
                if (!ReferenceEquals(rep, lastReplay)) { lastReplay = rep; log.Clear(); inFight = 0; }
                int a = Argmax(net.Forward(enc.B).Probs);
                bool fight = CombatManager.Instance.IsInProgress && driver.State.CurrentActIndex >= minAct
                             && driver.State.CurrentRoom is CombatRoom { RoomType: RoomType.Boss or RoomType.Elite } cr0
                             && (!bossOnly || cr0.RoomType == RoomType.Boss);
                if (fight) inFight++;
                if (saved == null && fight && inFight > after && rep != null)
                {
                    saved = Serialize(rep.serializableRun);
                    savedLog = log.ToList();
                    savedHash = enc.B.Hash();
                    where = $"floor {driver.State.TotalFloor} {(driver.State.CurrentRoom as CombatRoom)?.Encounter.Id.Entry}";
                    driver.AbortRequested = true;
                    return a;
                }
                log.Add(a);
                return a;
            };
            driver.StartRun(character, asc, runSeed);
            RunManager.Instance.CombatReplayWriter.IsEnabled = true; // on in the live game (off in TestMode)
            driver.PlayToEnd();
            string err1 = driver.Error?.Split('\n')[0] ?? "-";
            driver.EndRun();
            if (saved == null) { print($"run {r}: no capture (error {err1})"); continue; }

            int i = 0;
            ulong hash2 = 0;
            bool got = false;
            driver.Policy = (d, enc) =>
            {
                if (i < savedLog!.Count) return savedLog[i++];
                hash2 = enc.B.Hash();
                got = true;
                driver.AbortRequested = true;
                return 0;
            };
            if (Environment.GetEnvironmentVariable("SAVETEST_TRACE") == "1") driver.Trace = s => print("  T " + s.Split('\n')[0]);
            var save = Deserialize(saved);
            print($"  save: act {save.CurrentActIndex} visited {save.VisitedMapCoords.Count} last {(save.VisitedMapCoords.Count > 0 ? save.VisitedMapCoords[^1].ToString() : "-")} prefinished {save.PreFinishedRoom != null}");
            driver.StartFromSave(save);
            print($"  after load: room {driver.State.CurrentRoom?.GetType().Name} {(driver.State.CurrentRoom as CombatRoom)?.Encounter.Id.Entry} combat {CombatManager.Instance.IsInProgress} floor {driver.State.TotalFloor}");
            driver.PlayToEnd();
            string err2 = driver.Error?.Split('\n')[0] ?? "-";
            driver.EndRun();
            tested++;
            bool ok = got && hash2 == savedHash;
            if (ok) match++;
            print($"run {r} {where}: save {saved.Length} bytes, replayed {i}/{savedLog!.Count} decisions, {(ok ? "MATCH" : got ? "DIFFER" : "no decision reached")} (errors: {err1} / {err2})");
        }
        print($"matched {match}/{tested}");
    }
}

/// <summary>
/// End-to-end test of search as the mod uses it: plays greedy runs with the replay writer on, logs each room's
/// decisions, and at decisions in the selected fights whose top action has probability below --conf asks a
/// separately started search server (SearchClient, the same client as the mod) and plays its answer.
///
///   Sts2Sim.dll servertest --model m.bin --character ironclad --ascension 9 --seed 1001 --runs 10 [--rooms elite]
/// </summary>
public static class ServerTest
{
    static int Argmax(float[] p) { int b = 0; for (int i = 1; i < p.Length; i++) if (p[i] > p[b]) b = i; return b; }

    public static void Run(Func<string, string, string> arg, PumpContext pump, Vocab vocab, Action<string> print)
    {
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        string model = Path.GetFullPath(arg("--model", "model.bin"));
        var net = new PolicyNet(model);
        int runs = int.Parse(arg("--runs", "5")), k = int.Parse(arg("--k", "3")), n = int.Parse(arg("--n", "16"));
        float conf = float.Parse(arg("--conf", "0.95"), inv);
        string rooms = arg("--rooms", "elite");
        int character = RunDriver.CharacterIndex(arg("--character", "Ironclad")), asc = int.Parse(arg("--ascension", "0"));
        var rng = new Random(int.Parse(arg("--seed", "1")));
        string simDll = typeof(ServerTest).Assembly.Location;
        string dotnet = Environment.ProcessPath ?? "dotnet";
        using var client = new Sts2Ai.Shared.Search.SearchClient(dotnet, simDll);
        var driver = new RunDriver(pump, vocab);
        driver.Trace = s => { if (s.StartsWith("ERROR")) print("  " + s.Split('\n')[0]); };
        for (int r = 0; r < runs; r++)
        {
            ulong runSeed = (ulong)rng.NextInt64();
            object? lastReplay = null;
            var log = new List<int>();
            int searched = 0, changed = 0, desync = 0, errors = 0;
            double ms = 0;
            var sw = System.Diagnostics.Stopwatch.StartNew();
            driver.Policy = (d, enc) =>
            {
                var rep = Sts2Ai.Shared.Search.RoomState.Current();
                if (!ReferenceEquals(rep, lastReplay)) { lastReplay = rep; log.Clear(); }
                var probs = net.Forward(enc.B).Probs;
                int a = Argmax(probs);
                if (rep != null && probs[a] < conf && Searcher.InFight(driver, rooms))
                {
                    var resp = client.Search(new Sts2Ai.Shared.Search.SearchRequest
                    {
                        save = Convert.ToBase64String(Sts2Ai.Shared.Search.RoomState.Serialize(rep.serializableRun)),
                        decisions = log.ToArray(), model = model, hash = enc.B.Hash().ToString(), k = k, n = n,
                    }, TimeSpan.FromSeconds(120));
                    if (resp?.status == "ok") { searched++; ms += resp.ms; if (resp.action != a) changed++; a = resp.action; }
                    else if (resp?.status == "desync") desync++;
                    else { errors++; if (errors <= 3) print($"  search {resp?.status ?? "timeout"}: {resp?.msg}"); }
                }
                log.Add(a);
                return a;
            };
            driver.StartRun(character, asc, runSeed);
            RunManager.Instance.CombatReplayWriter.IsEnabled = true;
            driver.PlayToEnd();
            print($"run {r}: won={driver.Won} floor={driver.State.TotalFloor} act={driver.State.CurrentActIndex + 1} err={(driver.Error?.Split('\n')[0] ?? "-")} " +
                  $"searched={searched} changed={changed} desync={desync} errors={errors} search_ms={(searched > 0 ? ms / searched : 0):F0} minutes={sw.Elapsed.TotalMinutes:F1}");
            driver.EndRun();
        }
    }
}

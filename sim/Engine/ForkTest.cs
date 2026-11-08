using System.Diagnostics;
using System.Runtime.InteropServices;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Rooms;
using Sts2Ai.Shared.Encoding;
using Sts2Ai.Shared.Inference;

namespace Sts2Sim.Engine;

/// <summary>
/// Feasibility test for search by fork(): the simulator cannot clone a mid-run state, but the process can be
/// forked. Plays greedy runs; at the first decision of every boss fight it forks K children. Each child re-rolls
/// the dice (Diagnose.Reroll), plays the fight greedily and exits 1 if it won the fight, 0 if not. The parent
/// waits (timeout → the child is killed and counted as hung) and continues its own run.
///
///   Sts2Sim.dll forktest --model m.bin --character ironclad --ascension 10 --seed 1 --runs 2 --forks 16
/// </summary>
public static class ForkTest
{
    [DllImport("libc", SetLastError = true)] static extern int fork();
    [DllImport("libc", SetLastError = true)] static extern int waitpid(int pid, out int status, int options);
    [DllImport("libc", SetLastError = true)] static extern int kill(int pid, int sig);
    [DllImport("libc")] static extern void _exit(int status);
    const int WNOHANG = 1, SIGKILL = 9;

    static int Argmax(float[] p) { int b = 0; for (int i = 1; i < p.Length; i++) if (p[i] > p[b]) b = i; return b; }

    public static void Run(Func<string, string, string> arg, PumpContext pump, Vocab vocab, Action<string> print)
    {
        var net = new PolicyNet(arg("--model", "model.bin"));
        int runs = int.Parse(arg("--runs", "2")), k = int.Parse(arg("--forks", "16"));
        int character = RunDriver.CharacterIndex(arg("--character", "Ironclad")), asc = int.Parse(arg("--ascension", "0"));
        double timeout = double.Parse(arg("--timeout", "120"), System.Globalization.CultureInfo.InvariantCulture);
        var rng = new Random(int.Parse(arg("--seed", "1")));
        var driver = new RunDriver(pump, vocab);
        driver.Trace = s => { if (s.StartsWith("ERROR")) print("  " + s.Split('\n')[0]); };
        bool child = false;
        int totalForks = 0, hung = 0;
        for (int r = 0; r < runs; r++)
        {
            ulong runSeed = (ulong)rng.NextInt64();
            int lastFight = -1;
            driver.Policy = (d, enc) =>
            {
                bool boss = CombatManager.Instance.IsInProgress && driver.State.CurrentRoom is CombatRoom { RoomType: RoomType.Boss };
                if (!child && boss && driver.CombatsWon != lastFight)
                {
                    lastFight = driver.CombatsWon;
                    int cw = driver.CombatsWon, floor = driver.State.TotalFloor;
                    var sw = Stopwatch.StartNew();
                    Console.Out.Flush();
                    var pids = new List<int>();
                    bool dbg = Environment.GetEnvironmentVariable("FORK_DEBUG") == "1";
                    for (int j = 0; j < k; j++)
                    {
                        if (dbg) { Console.Error.WriteLine($"[parent {Environment.ProcessId}] fork {j}"); Console.Error.Flush(); }
                        int pid = fork();
                        if (dbg && pid != 0) { Console.Error.WriteLine($"[parent] forked {pid}"); Console.Error.Flush(); }
                        if (pid == 0)
                        {
                            // child: other dice from here on, greedy play; exit 1 once the fight is won
                            child = true;
                            PolicyNet.Sequential = true;
                            if (dbg) { Console.Error.WriteLine($"[child {j}] alive"); Console.Error.Flush(); }
                            Diagnose.Reroll(driver.State.Rng, $"fork{j}_{runSeed}_{floor}");
                            driver.Policy = (d2, e2) =>
                            {
                                if (driver.CombatsWon > cw || driver.State.TotalFloor != floor) _exit(1);
                                return Argmax(net.Forward(e2.B).Probs);
                            };
                            return Argmax(net.Forward(enc.B).Probs);
                        }
                        if (pid < 0) { print($"fork failed: errno {Marshal.GetLastWin32Error()}"); break; }
                        pids.Add(pid);
                    }
                    int wins = 0;
                    foreach (int pid in pids)
                    {
                        int st;
                        while (waitpid(pid, out st, WNOHANG) == 0)
                        {
                            if (sw.Elapsed.TotalSeconds > timeout) { kill(pid, SIGKILL); waitpid(pid, out st, 0); hung++; st = -1; break; }
                            Thread.Sleep(20);
                        }
                        if (st >= 0 && (st & 0x7f) == 0 && ((st >> 8) & 0xff) == 1) wins++;
                    }
                    totalForks += pids.Count;
                    print($"run {r} floor {floor} boss fight: greedy play with re-rolled dice won {wins}/{pids.Count} ({sw.Elapsed.TotalSeconds:F1}s)");
                }
                return Argmax(net.Forward(enc.B).Probs);
            };
            driver.StartRun(character, asc, runSeed);
            driver.PlayToEnd();
            if (child) _exit(driver.Won ? 1 : 0); // the fight ended the run: won it (1) or died (0)
            print($"run {r}: won={driver.Won} floor={driver.State.TotalFloor}");
            driver.EndRun();
        }
        print($"forks {totalForks}, hung {hung}");
    }

    static readonly System.Reflection.FieldInfo PileCards = typeof(MegaCrit.Sts2.Core.Entities.Cards.CardPile)
        .GetField("_cards", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
        ?? throw new MissingFieldException("CardPile._cards");

    /// <summary>Determinize the hidden state: the draw pile's order (unknown to the player) is reshuffled and
    /// every RNG stream re-rolled, so a rollout cannot use information the player does not have.</summary>
    static void Determinize(RunDriver driver, string key, int seed)
    {
        Diagnose.Reroll(driver.State.Rng, key);
        var draw = driver.Player.PlayerCombatState?.DrawPile;
        if (draw == null) return;
        var cards = (List<MegaCrit.Sts2.Core.Models.CardModel>)PileCards.GetValue(draw)!;
        var r = new Random(seed);
        for (int i = cards.Count - 1; i > 0; i--) { int j = r.Next(i + 1); (cards[i], cards[j]) = (cards[j], cards[i]); }
    }

    /// <summary>
    /// Greedy play with determinized rollout search in the selected fights (one-step policy improvement):
    /// at a decision whose top action has probability below --conf, each of the top --k actions (probability
    /// at least --min-p) is tried in --n forked children (sample j: the same determinization for every candidate). A child determinizes (Determinize), takes the action and
    /// plays the rest of the fight greedily; it exits 0 if the fight is lost, else 1 + HP%. The action with the
    /// best mean score (win counts most, then HP) is played. Prints one line per run like `play`.
    ///
    ///   Sts2Sim.dll searchplay --model m.bin --character ironclad --ascension 10 --seed 1 --runs 10 [--k 3 --n 8
    ///     --conf 0.95 --min-p 0.02 --rooms boss|elite|all]   (needs DOTNET_EnableWriteXorExecute=0)
    /// </summary>
    public static void SearchPlay(Func<string, string, string> arg, PumpContext pump, Vocab vocab, Action<string> print)
    {
        // fork() copies only the calling thread. W^X double-mapped code memory would be shared with the children
        // (crash); a thread-pool worker or the tiering JIT thread caught in managed code at fork time makes the
        // child's first GC wait for a thread that no longer exists (deadlock). So: no W^X, no tiered compilation,
        // and the network runs on the calling thread only, in the parent as well.
        if (Environment.GetEnvironmentVariable("DOTNET_EnableWriteXorExecute") != "0" || Environment.GetEnvironmentVariable("DOTNET_TieredCompilation") != "0")
            throw new InvalidOperationException("searchplay forks the process: set DOTNET_EnableWriteXorExecute=0 DOTNET_TieredCompilation=0");
        PolicyNet.Sequential = true;
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        var net = new PolicyNet(arg("--model", "model.bin"));
        int runs = int.Parse(arg("--runs", "10")), K = int.Parse(arg("--k", "3")), N = int.Parse(arg("--n", "16"));
        float conf = float.Parse(arg("--conf", "0.95"), inv), minP = float.Parse(arg("--min-p", "0.02"), inv);
        string rooms = arg("--rooms", "boss");
        int character = RunDriver.CharacterIndex(arg("--character", "Ironclad")), asc = int.Parse(arg("--ascension", "0"));
        double timeout = double.Parse(arg("--timeout", "300"), inv);
        var rng = new Random(int.Parse(arg("--seed", "1")));
        var driver = new RunDriver(pump, vocab);
        driver.Trace = s => { if (s.StartsWith("ERROR")) print("  " + s.Split('\n')[0]); };
        bool child = false;
        for (int r = 0; r < runs; r++)
        {
            ulong runSeed = (ulong)rng.NextInt64();
            int searched = 0, changed = 0, hung = 0, decisions = 0;
            var sw = Stopwatch.StartNew();
            int HpCode() => 1 + (int)Math.Round(100.0 * Math.Max(0, driver.Player.Creature.CurrentHp) / Math.Max(1, driver.Player.Creature.MaxHp));
            driver.Policy = (d, enc) =>
            {
                var probs = net.Forward(enc.B).Probs;
                int greedy = Argmax(probs);
                if (child) return greedy;
                decisions++;
                bool inCombat = CombatManager.Instance.IsInProgress;
                var room = driver.State.CurrentRoom as CombatRoom;
                bool want = inCombat && room != null && (rooms == "all"
                    || rooms == "boss" && room.RoomType == RoomType.Boss
                    || rooms == "elite" && room.RoomType is RoomType.Boss or RoomType.Elite);
                if (!want || probs[greedy] >= conf) return greedy;
                var cands = Enumerable.Range(0, probs.Length).Where(i => probs[i] >= minP)
                    .OrderByDescending(i => probs[i]).Take(K).ToList();
                if (cands.Count < 2) return greedy;
                int cw = driver.CombatsWon, floor = driver.State.TotalFloor;
                Console.Out.Flush();
                var pids = new List<(int pid, int cand)>();
                for (int c = 0; c < cands.Count; c++)
                    for (int j = 0; j < N; j++)
                    {
                        int pid = fork();
                        if (pid == 0)
                        {
                            child = true;
                            PolicyNet.Sequential = true;
                            // common random numbers: sample j uses the same determinization for every candidate, so
                            // candidates are compared on the same dice and draw orders (paired, much lower variance)
                            Determinize(driver, $"s{decisions}_{j}_{runSeed}", (int)((uint)runSeed * 31u + (uint)decisions * 7919u + (uint)j * 104729u));
                            driver.Policy = (d2, e2) =>
                            {
                                if (driver.CombatsWon > cw || driver.State.TotalFloor != floor) _exit(HpCode());
                                return Argmax(net.Forward(e2.B).Probs);
                            };
                            return cands[c];
                        }
                        if (pid < 0) throw new InvalidOperationException($"fork failed: errno {Marshal.GetLastWin32Error()}");
                        pids.Add((pid, c));
                    }
                var score = new double[cands.Count];
                var t0 = Stopwatch.StartNew();
                foreach (var (pid, c) in pids)
                {
                    int st;
                    while (waitpid(pid, out st, WNOHANG) == 0)
                    {
                        if (t0.Elapsed.TotalSeconds > timeout)
                        {
                            // SEARCH_KEEP_HUNG=1: leave the child for inspection (/proc/<pid>), report its pid
                            if (Environment.GetEnvironmentVariable("SEARCH_KEEP_HUNG") == "1") Console.Error.WriteLine($"HUNG {pid} cand {c} floor {floor}");
                            else { kill(pid, SIGKILL); waitpid(pid, out st, 0); }
                            hung++; st = -1; break;
                        }
                        Thread.Sleep(5);
                    }
                    if (st >= 0 && (st & 0x7f) == 0)
                    {
                        int code = (st >> 8) & 0xff;
                        if (code > 0) score[c] += 1.0 + 0.5 * (code - 1) / 100.0; // a win, plus up to 0.5 for HP left
                    }
                }
                int best = 0;
                for (int c = 1; c < cands.Count; c++)
                    if (score[c] > score[best] + 1e-9) best = c; // ties keep the policy's order
                searched++;
                if (cands[best] != greedy) changed++;
                return cands[best];
            };
            driver.StartRun(character, asc, runSeed);
            driver.PlayToEnd();
            if (child) _exit(driver.Won ? HpCode() : 0); // the fight ended the run
            print($"run {r}: won={driver.Won} floor={driver.State.TotalFloor} act={driver.State.CurrentActIndex + 1} steps={driver.Steps} " +
                  $"err={(driver.Error?.Split('\n')[0] ?? "-")} searched={searched} changed={changed} hung={hung} minutes={sw.Elapsed.TotalMinutes:F1}");
            driver.EndRun();
        }
    }
}

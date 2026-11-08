using System.Diagnostics;
using System.Runtime.InteropServices;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Rooms;
using Sts2Ai.Shared.Inference;

namespace Sts2Sim.Engine;

/// <summary>
/// Determinized rollout search by fork() (one-step policy improvement). At a decision, each of the top K actions
/// is tried in N forked children; child j of every candidate uses the same determinization (common random numbers):
/// the hidden draw-pile order is reshuffled and every RNG stream re-rolled, so no rollout sees information the
/// player does not have. A child takes its action, plays the rest of the fight greedily and exits 0 (fight lost)
/// or 1 + HP%. The parent returns the candidate with the best mean score (a win counts 1, plus up to 0.5 for HP).
///
/// Fork safety (all three were hit): no W^X double mapping (DOTNET_EnableWriteXorExecute=0), no tiered-compilation
/// background thread (DOTNET_TieredCompilation=0), and no thread-pool work (PolicyNet.Sequential), in the parent
/// as well, or a child's first GC waits for threads that do not exist after fork.
/// </summary>
public sealed class Searcher
{
    [DllImport("libc", SetLastError = true)] static extern int fork();
    [DllImport("libc", SetLastError = true)] static extern int waitpid(int pid, out int status, int options);
    [DllImport("libc", SetLastError = true)] static extern int kill(int pid, int sig);
    [DllImport("libc")] static extern void _exit(int status);
    const int WNOHANG = 1, SIGKILL = 9;

    public int K = 3, N = 16;
    public float MinP = 0.02f;
    public double Timeout = 120;
    public bool IsChild { get; private set; }
    public int Hung { get; private set; }
    readonly PolicyNet _net;

    public Searcher(PolicyNet net)
    {
        EnsureForkSafe();
        _net = net;
    }

    public static void EnsureForkSafe()
    {
        if (Environment.GetEnvironmentVariable("DOTNET_EnableWriteXorExecute") != "0" || Environment.GetEnvironmentVariable("DOTNET_TieredCompilation") != "0")
            throw new InvalidOperationException("search forks the process: set DOTNET_EnableWriteXorExecute=0 DOTNET_TieredCompilation=0");
        PolicyNet.Sequential = true;
    }

    static int Argmax(float[] p) { int b = 0; for (int i = 1; i < p.Length; i++) if (p[i] > p[b]) b = i; return b; }

    static readonly System.Reflection.FieldInfo PileCards = typeof(MegaCrit.Sts2.Core.Entities.Cards.CardPile)
        .GetField("_cards", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
        ?? throw new MissingFieldException("CardPile._cards");

    static void Determinize(RunDriver driver, string key, int seed)
    {
        Diagnose.Reroll(driver.State.Rng, key);
        var draw = driver.Player.PlayerCombatState?.DrawPile;
        if (draw == null) return;
        var cards = (List<MegaCrit.Sts2.Core.Models.CardModel>)PileCards.GetValue(draw)!;
        var r = new Random(seed);
        for (int i = cards.Count - 1; i > 0; i--) { int j = r.Next(i + 1); (cards[i], cards[j]) = (cards[j], cards[i]); }
    }

    int HpCode(RunDriver d) => 1 + (int)Math.Round(100.0 * Math.Max(0, d.Player.Creature.CurrentHp) / Math.Max(1, d.Player.Creature.MaxHp));

    /// <summary>Leave a forked child immediately (no runtime shutdown: it could wait for threads that do not exist).</summary>
    public static void ChildExit(int code) => _exit(code);

    /// <summary>A child whose run ended inside the fight (death or the final victory) reports and exits.</summary>
    public void ExitChild(RunDriver driver) => _exit(driver.Won ? HpCode(driver) : 0);

    /// <summary>The candidates searched at a decision: the top K actions with probability at least MinP.</summary>
    public List<int> Candidates(float[] probs) =>
        Enumerable.Range(0, probs.Length).Where(i => probs[i] >= MinP).OrderByDescending(i => probs[i]).Take(K).ToList();

    /// <summary>
    /// Search the current decision (call from the driver's Policy callback, return the result). In the parent it
    /// returns the best candidate; in a child (IsChild) it returns the candidate the child plays, after replacing
    /// driver.Policy with the greedy rollout.
    /// </summary>
    public (int action, List<int> cands, double[] scores) Search(RunDriver driver, float[] probs, uint key)
    {
        var cands = Candidates(probs);
        var scores = new double[cands.Count];
        if (cands.Count < 2) return (cands.Count == 1 ? cands[0] : Argmax(probs), cands, scores);
        int cw = driver.CombatsWon, floor = driver.State.TotalFloor;
        Console.Out.Flush();
        Console.Error.Flush();
        var pids = new List<(int pid, int cand)>();
        for (int c = 0; c < cands.Count; c++)
            for (int j = 0; j < N; j++)
            {
                int pid = fork();
                if (pid == 0)
                {
                    IsChild = true;
                    Console.SetOut(TextWriter.Null); // stdout may be a protocol channel
                    // common random numbers: sample j is the same determinization for every candidate
                    Determinize(driver, $"s{key}_{j}", (int)(key * 31u + (uint)j * 104729u));
                    driver.Policy = (d2, e2) =>
                    {
                        if (driver.CombatsWon > cw || driver.State.TotalFloor != floor) _exit(HpCode(driver));
                        return Argmax(_net.Forward(e2.B).Probs);
                    };
                    return (cands[c], cands, scores);
                }
                if (pid < 0) throw new InvalidOperationException($"fork failed: errno {Marshal.GetLastWin32Error()}");
                pids.Add((pid, c));
            }
        var t0 = Stopwatch.StartNew();
        foreach (var (pid, c) in pids)
        {
            int st;
            while (waitpid(pid, out st, WNOHANG) == 0)
            {
                if (t0.Elapsed.TotalSeconds > Timeout) { kill(pid, SIGKILL); waitpid(pid, out st, 0); Hung++; st = -1; break; }
                Thread.Sleep(2);
            }
            if (st >= 0 && (st & 0x7f) == 0)
            {
                int code = (st >> 8) & 0xff;
                if (code > 0) scores[c] += 1.0 + 0.5 * (code - 1) / 100.0;
            }
        }
        for (int c = 0; c < scores.Length; c++) scores[c] /= N;
        int best = 0;
        for (int c = 1; c < cands.Count; c++) if (scores[c] > scores[best] + 1e-9) best = c; // ties keep the policy's order
        return (cands[best], cands, scores);
    }

    /// <summary>Whether the current decision is in a fight of the given kind (boss | elite = boss or elite | all).</summary>
    public static bool InFight(RunDriver driver, string rooms) =>
        CombatManager.Instance.IsInProgress && driver.State.CurrentRoom is CombatRoom room && (rooms == "all"
            || rooms == "boss" && room.RoomType == RoomType.Boss
            || rooms == "elite" && room.RoomType is RoomType.Boss or RoomType.Elite);
}

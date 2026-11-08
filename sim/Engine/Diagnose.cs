using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using Sts2Ai.Shared.Encoding;
using Sts2Ai.Shared.Inference;

namespace Sts2Sim.Engine;

/// <summary>
/// Why does the policy lose? Plays greedy runs (the same runs as the `play` evaluation for the same --seed),
/// then for every loss replays the run exactly (the simulator is deterministic given seed + choices) and asks:
///   fight rescue: from the start of the fatal fight, does the sampled policy ever win that fight?
///                 (often = a tactical flaw; never = the run was already lost when the fight began)
///   act rescue:   from the start of that act, re-deciding everything (path, cards, rest, shop, fights),
///                 does it get past the fatal floor? (yes = an earlier strategic flaw)
///   run rescue:   re-playing the whole seed sampled, does it ever win? (never = a very hard seed)
///   luck test:    the same greedy play of the fatal fight, but the dice re-rolled just before it (all run RNG
///                 streams and the run seed replaced): how often does the *same policy* win that fight? High =
///                 the loss was mostly bad luck at this skill level; low = the fight is usually lost as played.
///                 A control replay without re-roll must reproduce the loss (determinism check).
/// Also prints one line per fight of the greedy run (encounter, HP in/out, outcome) for per-boss statistics.
///
///   Sts2Sim.dll diagnose --model m.bin --seed 701 --runs 10 [--from 0 --to 10] [--attempts 32] [--temp 1]
/// </summary>
public static class Diagnose
{
    sealed class Rec
    {
        public int Floor, Act, CombatsWon, Hp, MaxHp, Deck, Potions, Relics, Gold;
        public bool InCombat;
        public string Enc = "";
        public string RoomType = "";
        public float PWin;
    }

    public static void Run(Func<string, string, string> arg, PumpContext pump, Vocab vocab, Action<string> print)
    {
        var net = new PolicyNet(arg("--model", "model.bin"));
        int runs = int.Parse(arg("--runs", "10"));
        int seed = int.Parse(arg("--seed", "1"));
        int from = int.Parse(arg("--from", "0")), to = int.Parse(arg("--to", runs.ToString()));
        int character = RunDriver.CharacterIndex(arg("--character", "Ironclad"));
        int asc = int.Parse(arg("--ascension", "0"));
        int kFight = int.Parse(arg("--attempts", "32"));
        int kAct = int.Parse(arg("--act-attempts", kFight.ToString()));
        int kRun = int.Parse(arg("--run-attempts", "16"));
        int kLuck = int.Parse(arg("--luck-attempts", kFight.ToString()));
        float temp = float.Parse(arg("--temp", "1"));
        var rng = new Random(seed);
        var srng = new Random(seed * 7919 + 17);
        var driver = new RunDriver(pump, vocab);
        driver.Trace = s => { if (s.StartsWith("ERROR")) print("  " + s.Split('\n')[0]); };

        for (int r = 0; r < runs; r++)
        {
            ulong runSeed = (ulong)rng.NextInt64();
            if (r < from || r >= to) continue;

            // ---- greedy run, recorded
            var acts = new List<int>();
            var recs = new List<Rec>();
            driver.Policy = (d, enc) =>
            {
                var o = net.Forward(enc.B);
                int a = Argmax(o.Probs);
                acts.Add(a);
                recs.Add(Snapshot(driver, o.PWin));
                return a;
            };
            driver.StartRun(character, asc, runSeed);
            driver.PlayToEnd();
            bool won = driver.Won;
            int finalFloor = driver.State.TotalFloor, finalAct = driver.State.CurrentActIndex, finalCw = driver.CombatsWon;
            string cause = driver.Stalemate ? "stalemate" : driver.NoOpLoop ? "noop" : driver.Error != null ? "error" : "died";
            var finalRoom = driver.State.CurrentRoom;
            string fatalEnc = finalRoom is CombatRoom fcr ? fcr.Encounter.Id.Entry : finalRoom?.GetType().Name ?? "?";
            string fatalType = finalRoom is CombatRoom fcr2 ? fcr2.RoomType.ToString() : "-";
            driver.EndRun();

            // ---- per-fight lines of the greedy run
            for (int i = 0; i < recs.Count; i++)
            {
                if (!recs[i].InCombat || (i > 0 && recs[i - 1].InCombat && recs[i - 1].CombatsWon == recs[i].CombatsWon && recs[i - 1].Floor == recs[i].Floor)) continue;
                int j = i;
                while (j + 1 < recs.Count && recs[j + 1].InCombat && recs[j + 1].CombatsWon == recs[i].CombatsWon && recs[j + 1].Floor == recs[i].Floor) j++;
                bool fightWon = j + 1 < recs.Count || won;
                int hpOut = j + 1 < recs.Count ? recs[j + 1].Hp : (won ? -1 : 0);
                var s = recs[i];
                print($"fight run={r} floor={s.Floor} act={s.Act + 1} enc={s.Enc} type={s.RoomType} hp_in={s.Hp}/{s.MaxHp} hp_out={hpOut} won={fightWon} pwin_in={s.PWin:F3} deck={s.Deck} potions={s.Potions} relics={s.Relics}");
            }

            if (won)
            {
                print($"run {r} seed={runSeed}: won=True floor={finalFloor}");
                continue;
            }

            // ---- where the fatal fight starts in the decision sequence
            int s0 = recs.FindIndex(x => x.InCombat && x.Floor == finalFloor && x.CombatsWon == finalCw);
            int s1 = recs.FindIndex(x => x.Act == finalAct);
            var at = s0 >= 0 ? recs[s0] : null;
            string head = $"run {r} seed={runSeed}: won=False cause={cause} floor={finalFloor} act={finalAct + 1} enc={fatalEnc} type={fatalType}" +
                          (at != null ? $" hp_in={at.Hp}/{at.MaxHp} deck={at.Deck} potions={at.Potions} relics={at.Relics} gold={at.Gold} pwin_in={at.PWin:F3} pwin_act={recs[Math.Max(0, s1)].PWin:F3}" : "");
            if (s0 < 0 || cause != "died")
            {
                print(head + " (no rescue: not a fight death)");
                continue;
            }

            // ---- fight rescue: replay to the fatal fight, then sample only inside it
            int fightWins = 0, diverged = 0;
            var hpLeft = new List<int>();
            for (int k = 0; k < kFight; k++)
            {
                var res = Attempt(driver, net, acts, s0, temp, srng, runSeed, character, asc,
                    stop: dv => dv.CombatsWon > finalCw || dv.State.TotalFloor != finalFloor,
                    success: dv => dv.CombatsWon > finalCw,
                    check: dv => dv.State.TotalFloor == finalFloor && CombatManager.Instance.IsInProgress);
                if (res.Diverged) diverged++;
                if (res.Success) { fightWins++; hpLeft.Add(res.Hp); }
            }

            // ---- luck test: greedy play of the fatal fight with re-rolled dice (+ one control without re-roll)
            int luckWins = 0;
            bool controlWon = false;
            if (s0 > 0 && kLuck > 0)
            {
                for (int k = 0; k <= kLuck; k++)
                {
                    var res = Attempt(driver, net, acts, s0, temp, srng, runSeed, character, asc,
                        stop: dv => dv.CombatsWon > finalCw || dv.State.TotalFloor != finalFloor,
                        success: dv => dv.CombatsWon > finalCw,
                        check: dv => dv.State.TotalFloor == finalFloor && CombatManager.Instance.IsInProgress,
                        rerollAt: k == 0 ? -1 : s0 - 1, rerollKey: k, greedy: true);
                    if (res.Diverged) diverged++;
                    if (k == 0) controlWon = res.Success;
                    else if (res.Success) luckWins++;
                }
            }

            // ---- act rescue: replay to the start of the act, then re-decide everything until past the fatal floor
            int actWins = 0;
            if (s1 >= 0)
                for (int k = 0; k < kAct; k++)
                {
                    var res = Attempt(driver, net, acts, s1, temp, srng, runSeed, character, asc,
                        stop: dv => dv.State.TotalFloor > finalFloor,
                        success: dv => dv.State.TotalFloor > finalFloor || dv.Won,
                        check: dv => dv.State.CurrentActIndex == finalAct);
                    if (res.Diverged) diverged++;
                    if (res.Success) actWins++;
                }

            // ---- run rescue (only when neither rescue worked): is the seed ever won when sampled from the start?
            string runPart = "";
            if (fightWins == 0 && actWins == 0 && kRun > 0)
            {
                int runWins = 0;
                for (int k = 0; k < kRun; k++)
                {
                    var res = Attempt(driver, net, acts, 0, temp, srng, runSeed, character, asc,
                        stop: _ => false, success: dv => dv.Won, check: _ => true);
                    if (res.Success) runWins++;
                }
                runPart = $" rescue_run={runWins}/{kRun}";
            }
            string hpPart = hpLeft.Count > 0 ? $" hp_after_rescue={hpLeft.Average():F0}" : "";
            string luckPart = kLuck > 0 && s0 > 0 ? $" luck_greedy={luckWins}/{kLuck}" + (controlWon ? " CONTROL_WON" : "") : "";
            print(head + $" rescue_fight={fightWins}/{kFight} rescue_act={actWins}/{kAct}{runPart}{hpPart}{luckPart}" + (diverged > 0 ? $" DIVERGED={diverged}" : ""));
        }
    }

    readonly record struct Result(bool Success, bool Diverged, int Hp);

    /// <summary>Replays the recorded choices for the first <paramref name="prefix"/> policy calls, then samples.</summary>
    static Result Attempt(RunDriver driver, PolicyNet net, List<int> acts, int prefix, float temp, Random srng,
        ulong runSeed, int character, int asc, Func<RunDriver, bool> stop, Func<RunDriver, bool> success, Func<RunDriver, bool> check,
        int rerollAt = -1, int rerollKey = 0, bool greedy = false)
    {
        int calls = 0;
        bool diverged = false, done = false, ok = false;
        int hp = 0;
        driver.Policy = (d, enc) =>
        {
            int i = calls++;
            if (done) return 0;
            if (i < prefix)
            {
                if (acts[i] >= d.Actions.Count) { diverged = true; done = true; driver.AbortRequested = true; return 0; }
                if (i == rerollAt) Reroll(driver.State.Rng, $"reroll{rerollKey}_{runSeed}");
                return acts[i];
            }
            if (i == prefix && !check(driver)) { diverged = true; done = true; driver.AbortRequested = true; return 0; }
            if (stop(driver))
            {
                ok = success(driver);
                hp = driver.Player.Creature.CurrentHp;
                done = true;
                driver.AbortRequested = true;
                return 0;
            }
            var probs = net.Forward(enc.B).Probs;
            return greedy ? Argmax(probs) : Sample(probs, temp, srng);
        };
        driver.StartRun(character, asc, runSeed);
        driver.PlayToEnd();
        if (!done)
        {
            // the run ended on its own (death, victory) before the stop condition was seen at a decision
            ok = !diverged && success(driver);
            hp = driver.Player.Creature.CurrentHp;
        }
        driver.EndRun();
        return new Result(ok && !diverged, diverged, hp);
    }

    static readonly System.Reflection.FieldInfo RngsField = typeof(RunRngSet).GetField("_rngs",
        System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic) ?? throw new MissingFieldException("RunRngSet._rngs");
    static readonly System.Reflection.FieldInfo SeedField = typeof(RunRngSet).GetField("<Seed>k__BackingField",
        System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic) ?? throw new MissingFieldException("RunRngSet.Seed");

    /// <summary>Re-roll the dice: every run RNG stream restarts from a new seed, and the run seed (from which
    /// per-entity Rngs created later are derived) changes too.</summary>
    internal static void Reroll(RunRngSet set, string newSeed)
    {
        var fresh = new RunRngSet(newSeed);
        var dst = (System.Collections.IDictionary)RngsField.GetValue(set)!;
        var src = (System.Collections.IDictionary)RngsField.GetValue(fresh)!;
        foreach (System.Collections.DictionaryEntry e in src) dst[e.Key] = e.Value;
        SeedField.SetValue(set, fresh.Seed);
    }

    static Rec Snapshot(RunDriver dv, float pwin)
    {
        var p = dv.Player;
        var room = dv.State.CurrentRoom;
        bool inCombat = CombatManager.Instance.IsInProgress;
        return new Rec
        {
            Floor = dv.State.TotalFloor,
            Act = dv.State.CurrentActIndex,
            CombatsWon = dv.CombatsWon,
            Hp = p.Creature.CurrentHp,
            MaxHp = p.Creature.MaxHp,
            Deck = p.Deck.Cards.Count,
            Potions = p.Potions.Count(),
            Relics = p.Relics.Count,
            Gold = p.Gold,
            InCombat = inCombat,
            Enc = inCombat && room is CombatRoom cr ? cr.Encounter.Id.Entry : "",
            RoomType = inCombat && room is CombatRoom cr2 ? cr2.RoomType.ToString() : "",
            PWin = pwin,
        };
    }

    static int Argmax(float[] p)
    {
        int k = 0;
        for (int i = 1; i < p.Length; i++) if (p[i] > p[k]) k = i;
        return k;
    }

    static int Sample(float[] p, float temp, Random rng)
    {
        double total = 0;
        var w = new double[p.Length];
        for (int i = 0; i < p.Length; i++) { w[i] = temp == 1f ? p[i] : Math.Pow(Math.Max(p[i], 1e-12), 1.0 / temp); total += w[i]; }
        double u = rng.NextDouble() * total, c = 0;
        for (int i = 0; i < p.Length; i++) { c += w[i]; if (u <= c) return i; }
        return p.Length - 1;
    }
}

using System.Diagnostics;
using Sts2Ai.Shared.Encoding;

namespace Sts2Sim.Engine;

/// <summary>
/// CLI entry point (modes: random, worker, vocab, mapcheck).
/// </summary>
public static class SimMain
{
    public static void Run(string[] args, Action<string> print, string instanceTag)
    {
        string mode = args.Length > 0 ? args[0] : "random";
        bool quiet = args.Contains("--quiet");
        string Arg(string name, string def) { int i = Array.IndexOf(args, name); return i >= 0 && i + 1 < args.Length ? args[i + 1] : def; }

        SimBoot.Init(quiet, instanceTag);
        var pump = PumpContext.Install();
        var vocab = Vocab.FromModelDb();
        string seedProtocol = Arg("--seed-protocol", "legacy");
        if (seedProtocol is not ("legacy" or "canonical")) throw new ArgumentException("unknown seed protocol");
        RunDriver.CanonicalSeeds = seedProtocol == "canonical";

        FastMapPruning.Verify = args.Contains("--verify-maps");
        StateEncoder.EncodeAllies = args.Contains("--allies");
        switch (mode)
        {
            case "mapcheck":
            {
                // generate every act's map for many seeds (with --verify-maps: compare against the original algorithm)
                int n = int.Parse(Arg("--runs", "200"));
                var driver = new RunDriver(pump, vocab);
                var sw = Stopwatch.StartNew();
                int maps = 0;
                for (int r = 0; r < n; r++)
                {
                    driver.StartRun(r % 5, 0, (ulong)(r * 7919 + 13));
                    var st = driver.State;
                    for (int act = 1; act < st.Acts.Count; act++)
                    {
                        st.CurrentActIndex = act;
                        st.Act.CreateMap(st, replaceTreasureWithElites: false);
                        maps++;
                    }
                    st.CurrentActIndex = 0;
                    driver.EndRun();
                    maps++;
                }
                print($"maps={maps} verified={FastMapPruning.Verified} time={sw.Elapsed.TotalSeconds:F1}s");
                break;
            }
            case "dumpobs":
            {
                // record encoded observations from random play (inputs for the PyTorch <-> C# parity test)
                int want = int.Parse(Arg("--count", "300"));
                var rng = new Random(int.Parse(Arg("--seed", "1")));
                using var bw = new BinaryWriter(File.Create(Arg("--out", "obs.bin")));
                bw.Write("STS2OBS1".ToCharArray());
                bw.Write(want);
                int got = 0;
                var driver = new RunDriver(pump, vocab);
                driver.Policy = (d, enc) =>
                {
                    if (got < want && rng.NextDouble() < 0.2)
                    {
                        var b = enc.B;
                        bw.Write(b.NumTokens); bw.Write(b.NumActions);
                        for (int i = 0; i < b.NumTokens * ObsSpec.NTokCat; i++) bw.Write(b.TokCat[i]);
                        for (int i = 0; i < b.NumTokens * ObsSpec.FTok; i++) bw.Write(b.TokNum[i]);
                        for (int i = 0; i < b.NumActions * ObsSpec.NActCat; i++) bw.Write(b.ActCat[i]);
                        for (int i = 0; i < b.NumActions * 2; i++) bw.Write(b.ActRef[i]);
                        for (int i = 0; i < b.NumActions * ObsSpec.FAct; i++) bw.Write(b.ActNum[i]);
                        got++;
                    }
                    return rng.Next(d.Actions.Count);
                };
                for (int r = 0; got < want && r < 10000; r++)
                {
                    driver.StartRun(r % 5, 0, (ulong)rng.NextInt64());
                    driver.PlayToEnd();
                    driver.EndRun();
                }
                print($"wrote {got} observations");
                break;
            }
            case "play":
            {
                // play full runs with the C# PolicyNet (what the mod runs): greedy, or sampled with --sample
                var net = new Sts2Ai.Shared.Inference.PolicyNet(Arg("--model", "model.bin"));
                StateEncoder.EncodeAllies |= net.EncodeAllies;
                int runs = int.Parse(Arg("--runs", "10"));
                if ((args.Contains("--runseed") || args.Contains("--game-seed")) && runs != 1)
                    throw new ArgumentException("explicit seed requires --runs 1");
                bool sample = args.Contains("--sample");
                var rng = new Random(int.Parse(Arg("--seed", "1")));
                int character = RunDriver.CharacterIndex(Arg("--character", "Ironclad"));
                int[]? coopPlay = Arg("--coop", "") is { Length: > 0 } ccs ? ccs.Split(',').Select(RunDriver.CharacterIndex).ToArray() : null;
                var driver = new RunDriver(pump, vocab);
                driver.Trace = s => { if (s.StartsWith("ERROR")) print("  " + s.Split('\n')[0]); };
                var stats = new Dictionary<string, int>();
                void Count(string k) => stats[k] = stats.GetValueOrDefault(k) + 1;
                bool traceAll = args.Contains("--trace");
                bool forceRemove = args.Contains("--force-remove");
                bool removeBasic = args.Contains("--remove-basic");
                bool removing = false;
                driver.Policy = (d, enc) =>
                {
                    var o = net.Forward(enc.B);
                    if (traceAll && d.Type != DecisionType.Combat)
                    {
                        var st = driver.State;
                        var top = Enumerable.Range(0, d.Actions.Count).OrderByDescending(i => o.Probs[i]).Take(3)
                            .Select(i => $"{d.Actions[i]} {o.Probs[i] * 100:F0}%");
                        print($"  F{st.TotalFloor} hp {driver.Player.Creature.CurrentHp}/{driver.Player.Creature.MaxHp} gold {driver.Player.Gold} deck {driver.Player.Deck.Cards.Count} P(win) {o.PWin * 100:F1}% [{d.Type}] " + string.Join(" | ", top));
                    }
                    int pick;
                    if (!sample) { pick = 0; for (int i = 1; i < o.Probs.Length; i++) if (o.Probs[i] > o.Probs[pick]) pick = i; }
                    else
                    {
                        double u = rng.NextDouble(), c = 0;
                        pick = o.Probs.Length - 1;
                        for (int i = 0; i < o.Probs.Length; i++) { c += o.Probs[i]; if (u <= c) { pick = i; break; } }
                    }
                    // counterfactual test: always buy card removal when it is offered (the policy picks the card)
                    if (forceRemove && d.Type == DecisionType.Shop)
                    {
                        int rmi = d.Actions.FindIndex(x => x.Type == ActionType.RemoveCardService);
                        if (rmi >= 0) { pick = rmi; removing = true; }
                    }
                    else if (removing && d.Type == DecisionType.CardSelect)
                    {
                        // which card the removal takes: the standard heuristic (a basic Strike, then a basic Defend)
                        removing = false;
                        if (removeBasic)
                        {
                            int bi = d.Actions.FindIndex(x => x.Label.StartsWith("select STRIKE_"));
                            if (bi < 0) bi = d.Actions.FindIndex(x => x.Label.StartsWith("select DEFEND_"));
                            if (bi >= 0) pick = bi;
                        }
                    }
                    // observation size: how often the token budget is (nearly) exhausted, per act
                    {
                        string ak = d.Type == DecisionType.Combat ? $"tok.combat.act{driver.State.CurrentActIndex + 1}" : $"tok.other.act{driver.State.CurrentActIndex + 1}";
                        Count(ak + ".n");
                        // decisions per room kind and act: how much of the training data each kind of fight gets
                        var rk = driver.State.CurrentRoom is MegaCrit.Sts2.Core.Rooms.CombatRoom cr ? cr.RoomType.ToString() : "NonCombat";
                        Count($"room.act{driver.State.CurrentActIndex + 1}.{(d.Type == DecisionType.Combat ? rk : "NonCombat")}");
                        stats[ak + ".sum"] = stats.GetValueOrDefault(ak + ".sum") + enc.B.NumTokens;
                        if (enc.B.TokenCapacityLeft <= 4) Count(ak + ".full");
                        if (enc.B.DroppedTokens > 0) Count(ak + ".dropped");
                    }
                    // turn endings that leave energy and playable cards unused (per room type)
                    if (d.Type == DecisionType.Combat && d.Actions[pick].Type == ActionType.EndTurn)
                    {
                        var rt = (driver.State.CurrentRoom as MegaCrit.Sts2.Core.Rooms.CombatRoom)?.RoomType.ToString() ?? "x";
                        Count($"endturn.{rt}");
                        int energy = driver.Player.PlayerCombatState?.Energy ?? 0;
                        if (energy > 0 && d.Actions.Any(x => x.Type == ActionType.PlayCard && x.Num[0] * 3 <= energy + 0.01f))
                            Count($"endturn.{rt}.unused");
                    }
                    // strategic behaviour summary
                    var a = d.Actions[pick];
                    switch (d.Type)
                    {
                        case DecisionType.CardReward:
                            Count(a.Type == ActionType.SelectCard ? "cardreward.take" : "cardreward.skip");
                            Count("cardreward.skip.p%." + (int)(o.Probs.Where((_, i) => d.Actions[i].Type == ActionType.Skip).Sum() * 100 / 5 * 5));
                            break;
                        case DecisionType.Rest: Count("rest." + a.Label.Replace("rest ", "")); break;
                        case DecisionType.Map: if (a.Type == ActionType.ChooseMapNode) Count("map." + ((MegaCrit.Sts2.Core.Map.MapPoint)a.Payload!).PointType); else if (a.Type == ActionType.UsePotion) Count("potion.outofcombat"); break;
                        case DecisionType.Shop:
                            Count("shop." + a.Type);
                            foreach (var t in d.Actions.Select(x => x.Type).Distinct()) Count("shop.offered." + t);
                            if (d.Actions.FirstOrDefault(x => x.Type == ActionType.RemoveCardService) is { } rm)
                            { Count("shop.remove.p%." + (int)(o.Probs[d.Actions.IndexOf(rm)] * 100 / 5 * 5)); }
                            break;

                        case DecisionType.Treasure: Count("treasure." + a.Type); break;
                        case DecisionType.Combat: if (a.Type == ActionType.UsePotion) Count("potion.combat"); break;
                    }
                    return pick;
                };
                int wins = 0; double floors = 0;
                var sw = Stopwatch.StartNew();
                for (int r = 0; r < runs; r++)
                {
                    ulong runSeed = args.Contains("--runseed") ? ulong.Parse(Arg("--runseed", "0")) : (ulong)rng.NextInt64();
                    if (args.Contains("--game-seed"))
                    {
                        if (coopPlay != null) throw new ArgumentException("string-seed CLI currently supports solo only");
                        driver.StartRun(character, int.Parse(Arg("--ascension", "0")), Arg("--game-seed", ""));
                    }
                    else driver.StartRun(coopPlay ?? new[] { character }, int.Parse(Arg("--ascension", "0")), runSeed);
                    driver.PlayToEnd();
                    if (driver.Won) wins++;
                    floors += driver.State.TotalFloor;
                    stats["end.deck"] = stats.GetValueOrDefault("end.deck") + driver.Player.Deck.Cards.Count;
                    stats["end.gold"] = stats.GetValueOrDefault("end.gold") + driver.Player.Gold;
                    stats["end.potions_held"] = stats.GetValueOrDefault("end.potions_held") + driver.Player.Potions.Count();
                    stats["end.relics"] = stats.GetValueOrDefault("end.relics") + driver.Player.Relics.Count;
                    print($"run {r}: won={driver.Won} floor={driver.State.TotalFloor} act={driver.State.CurrentActIndex + 1} steps={driver.Steps} err={(driver.Error?.Split('\n')[0] ?? "-")} " +
                          $"potions_end={driver.Player.Potions.Count()} potion_ids={string.Join(",", driver.Player.Potions.Select(p => p.Id.Entry))} " +
                          $"cause={(driver.Won ? "won" : driver.Stalemate ? "stalemate" : driver.NoOpLoop ? "noop" : driver.Error != null ? "error" : "died")} " +
                          $"room={(driver.State.CurrentRoom as MegaCrit.Sts2.Core.Rooms.CombatRoom)?.Encounter.Id.Entry ?? driver.State.CurrentRoom?.GetType().Name} seed={runSeed} seed_string={driver.SeedString} seed_protocol={driver.SeedProtocol}");
                    driver.EndRun();
                }
                print($"play: runs={runs} wins={wins} avgFloor={floors / runs:F1} time={sw.Elapsed.TotalSeconds:F0}s");
                print("stats " + string.Join(" ", stats.OrderBy(k => k.Key).Select(k => $"{k.Key}={k.Value}")));
                break;
            }
            case "parity":
            {
                // compare the C# PolicyNet with PyTorch outputs written by train/sts2ai/export.py
                var net = new Sts2Ai.Shared.Inference.PolicyNet(Arg("--model", "model.bin"));
                using var obs = new BinaryReader(File.OpenRead(Arg("--obs", "obs.bin")));
                using var exp = new BinaryReader(File.OpenRead(Arg("--expect", "parity.bin")));
                if (new string(obs.ReadChars(8)) != "STS2OBS1" || new string(exp.ReadChars(8)) != "STS2PAR1") throw new InvalidDataException("bad files");
                int n = obs.ReadInt32();
                if (exp.ReadInt32() != n) throw new InvalidDataException("case count mismatch");
                double maxLogit = 0, maxV = 0, maxP = 0; int argmaxAgree = 0;
                var sw = Stopwatch.StartNew();
                for (int c = 0; c < n; c++)
                {
                    int nt = obs.ReadInt32(), na = obs.ReadInt32();
                    int[] RI(int k) { var x = new int[k]; for (int i = 0; i < k; i++) x[i] = obs.ReadInt32(); return x; }
                    float[] RF(BinaryReader r, int k) { var x = new float[k]; for (int i = 0; i < k; i++) x[i] = r.ReadSingle(); return x; }
                    var tc = RI(nt * ObsSpec.NTokCat); var tn = RF(obs, nt * ObsSpec.FTok);
                    var ac = RI(na * ObsSpec.NActCat); var ar = RI(na * 2); var an = RF(obs, na * ObsSpec.FAct);
                    if (exp.ReadInt32() != na) throw new InvalidDataException("action count mismatch");
                    var el = RF(exp, na); var ev = RF(exp, 4);
                    var o = net.Forward(nt, tc, tn, na, ac, ar, an);
                    for (int k = 0; k < na; k++) maxLogit = Math.Max(maxLogit, Math.Abs(o.Logits[k] - el[k]));
                    maxV = Math.Max(maxV, Math.Max(Math.Abs(o.VReturn - ev[0]), Math.Max(Math.Abs(o.Floor - ev[2]), Math.Abs(o.Hp - ev[3]))));
                    maxP = Math.Max(maxP, Math.Abs(o.PWin - ev[1]));
                    if (Array.IndexOf(o.Logits, o.Logits.Max()) == Array.IndexOf(el, el.Max())) argmaxAgree++;
                }
                print($"parity: cases={n} max|dlogit|={maxLogit:E2} max|dvalue|={maxV:E2} max|dPwin|={maxP:E2} argmax agree={argmaxAgree}/{n} " +
                      $"C# time/decision={sw.Elapsed.TotalMilliseconds / n:F1}ms");
                break;
            }
            case "vocab":
                vocab.Save(Arg("--out", "vocab.json"), "v0.107.1");
                print($"vocab size {vocab.Size} ({vocab.Names.Count} ids)");
                break;
            case "text":
                TextSession.Run(Arg, pump, vocab, print);
                break;
            case "llm":
            {
                // benchmark a general LLM as the policy (Anthropic-compatible /v1/messages endpoint)
                var llm = new LlmPolicy(Arg("--endpoint", ""), Arg("--llm-model", ""), Arg("--key-file", ""), int.Parse(Arg("--max-tokens", "400")));
                string traceFile = Arg("--trace-file", "");
                StreamWriter? tw = traceFile.Length > 0 ? new StreamWriter(traceFile) { AutoFlush = true } : null;
                if (tw != null) llm.Log = s => tw.WriteLine(s);
                int runs = int.Parse(Arg("--runs", "1"));
                var rng = new Random(int.Parse(Arg("--seed", "1")));
                int character = RunDriver.CharacterIndex(Arg("--character", "Ironclad"));
                var driver = new RunDriver(pump, vocab);
                driver.Trace = s => { if (s.StartsWith("ERROR")) print("  " + s.Split('\n')[0]); };
                driver.Policy = (d, enc) => llm.Choose(driver, d);
                for (int r = 0; r < runs; r++)
                {
                    var sw = Stopwatch.StartNew();
                    driver.StartRun(character, int.Parse(Arg("--ascension", "0")), (ulong)rng.NextInt64());
                    driver.PlayToEnd();
                    print($"run {r}: won={driver.Won} floor={driver.State.TotalFloor} act={driver.State.CurrentActIndex + 1} steps={driver.Steps} err={(driver.Error?.Split('\n')[0] ?? "-")} " +
                          $"cause={(driver.Won ? "won" : driver.Stalemate ? "stalemate" : driver.NoOpLoop ? "noop" : driver.Error != null ? "error" : "died")} " +
                          $"room={(driver.State.CurrentRoom as MegaCrit.Sts2.Core.Rooms.CombatRoom)?.Encounter.Id.Entry ?? driver.State.CurrentRoom?.GetType().Name} " +
                          $"llm_calls={llm.Calls} invalid={llm.Invalid} retries={llm.Retries} sec_per_call={(llm.Calls > 0 ? llm.Seconds / llm.Calls : 0):F2} in_tok={llm.InTokens} out_tok={llm.OutTokens} minutes={sw.Elapsed.TotalMinutes:F0}");
                    driver.EndRun();
                }
                tw?.Dispose();
                break;
            }
            case "diagnose":
                Diagnose.Run(Arg, pump, vocab, print);
                break;
            case "forktest":
                ForkTest.Run(Arg, pump, vocab, print);
                break;
            case "searchplay":
                ForkTest.SearchPlay(Arg, pump, vocab, print);
                break;
            case "savetest":
                SaveTest.Run(Arg, pump, vocab, print);
                break;
            case "servertest":
                ServerTest.Run(Arg, pump, vocab, print);
                break;
            case "searchserver":
                SearchServer.Run(Arg, pump, vocab, print);
                break;
            case "worker":
                new Worker(pump, vocab, Arg("--slot", ""), int.Parse(Arg("--ascension", "0")), Arg("--characters", "Ironclad"),
                    double.Parse(Arg("--restart-prob", "0"), System.Globalization.CultureInfo.InvariantCulture),
                    double.Parse(Arg("--coop-prob", "0"), System.Globalization.CultureInfo.InvariantCulture),
                    int.Parse(Arg("--coop-max-players", "3")), Arg("--practice-sampler", "uncertainty"),
                    boss2Situation: Arg("--boss2-situation", "0") == "1", boss2HpStratified: Arg("--boss2-hp-stratified", "0") == "1",
                    fightOutcome: Arg("--fight-outcome", "0") == "1", practiceRepeats: int.Parse(Arg("--practice-repeats", "1")),
                    practiceTrace: Arg("--practice-trace", "0") == "1").Run();
                break;
            default:
            {
                int runs = int.Parse(Arg("--runs", "10"));
                bool trace = args.Contains("--trace");
                var rng = new Random(int.Parse(Arg("--seed", "1")));
                var driver = new RunDriver(pump, vocab);
                if (trace) driver.Trace = s => print("  " + s);
                else driver.Trace = s => { if (s.StartsWith("ERROR")) print("  " + s.Split('\n')[0]); };
                driver.GodMode = args.Contains("--god");
                int character = RunDriver.CharacterIndex(Arg("--character", "Ironclad"));
                // --coop ironclad,silent[,...]: a co-op run with one player per listed character
                int[]? coop = Arg("--coop", "") is { Length: > 0 } cs ? cs.Split(',').Select(RunDriver.CharacterIndex).ToArray() : null;
                driver.Policy = (d, enc) =>
                {
                    // random policy biased toward playing cards before ending turn
                    if (d.Type == Sts2Ai.Shared.Encoding.DecisionType.Combat && d.Actions.Count > 1 && rng.NextDouble() < 0.85)
                    {
                        var plays = Enumerable.Range(0, d.Actions.Count).Where(i => d.Actions[i].Type == Sts2Ai.Shared.Encoding.ActionType.PlayCard).ToList();
                        if (plays.Count > 0) return plays[rng.Next(plays.Count)];
                    }
                    return rng.Next(d.Actions.Count);
                };
                var sw = Stopwatch.StartNew();
                Stopwatch tStart = new(), tPlay = new(), tEnd = new();
                long steps = 0; int wins = 0, errors = 0; double floors = 0;
                var errs = new Dictionary<string, int>();
                for (int r = 0; r < runs; r++)
                {
                    ulong runSeed = args.Contains("--runseed") ? ulong.Parse(Arg("--runseed", "0")) : (ulong)rng.NextInt64();
                    tStart.Start(); driver.StartRun(coop ?? new[] { character }, int.Parse(Arg("--ascension", "0")), runSeed); tStart.Stop();
                    tPlay.Start(); driver.PlayToEnd(); tPlay.Stop();
                    steps += driver.Steps;
                    floors += driver.State.TotalFloor;
                    if (driver.Won) wins++;
                    if (driver.Error != null) { errors++; var k = driver.Error.Split('\n')[0]; errs[k] = errs.GetValueOrDefault(k) + 1; }
                    print($"run {r} seed={runSeed}: won={driver.Won} floor={driver.State.TotalFloor} act={driver.State.CurrentActIndex + 1} steps={driver.Steps} combatsWon={driver.CombatsWon} err={(driver.Error?.Split('\n')[0] ?? "-")}");
                    tEnd.Start(); driver.EndRun(); tEnd.Stop();
                }
                print($"time: start={tStart.Elapsed.TotalSeconds:F1}s play={tPlay.Elapsed.TotalSeconds:F1}s (waiting {driver.WaitTime.Elapsed.TotalSeconds:F1}s in {driver.Waits} waits) end={tEnd.Elapsed.TotalSeconds:F1}s");
                print($"runs={runs} wins={wins} errors={errors} avgFloor={floors / runs:F1} steps={steps} time={sw.Elapsed.TotalSeconds:F1}s steps/s={steps / sw.Elapsed.TotalSeconds:F0}");
                foreach (var kv in errs.OrderByDescending(k => k.Value)) print($"  {kv.Value}x {kv.Key}");
                if (args.Contains("--meminfo")) print(MemInfo());
                break;
            }
        }

    }

    public static string MemInfo()
    {
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        var gi = GC.GetGCMemoryInfo();
        var proc = Process.GetCurrentProcess();
        return $"mem: rss={proc.WorkingSet64 >> 20}MB private={proc.PrivateMemorySize64 >> 20}MB gcHeapLive={GC.GetTotalMemory(false) >> 20}MB " +
               $"gcCommitted={gi.TotalCommittedBytes >> 20}MB jitMethods={System.Runtime.JitInfo.GetCompiledMethodCount()} jitIL={System.Runtime.JitInfo.GetCompiledILBytes() >> 10}KB " +
               $"assemblies={AppDomain.CurrentDomain.GetAssemblies().Length}";
    }
}

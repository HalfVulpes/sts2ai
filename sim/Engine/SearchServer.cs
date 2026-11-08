using System.Diagnostics;
using System.Text.Json;
using Sts2Ai.Shared.Encoding;
using Sts2Ai.Shared.Inference;
using Sts2Ai.Shared.Search;

namespace Sts2Sim.Engine;

/// <summary>
/// `Sts2Sim.dll searchserver`: search for the in-game mod (and `servertest`). Reads one SearchRequest per line on
/// stdin; for each, restores the room from its room-entry state (RunDriver.StartFromSave), replays the room's
/// decisions, checks that the current observation hash equals the client's (else "desync"), searches the
/// decision (Searcher) and writes one SearchResponse line to stdout. Needs the fork-safety environment
/// (DOTNET_EnableWriteXorExecute=0 DOTNET_TieredCompilation=0); SearchClient sets it.
/// </summary>
public static class SearchServer
{
    static int Argmax(float[] p) { int b = 0; for (int i = 1; i < p.Length; i++) if (p[i] > p[b]) b = i; return b; }

    public static void Run(Func<string, string, string> arg, PumpContext pump, Vocab vocab, Action<string> print)
    {
        Searcher.EnsureForkSafe();
        var stdout = Console.Out;
        var nets = new Dictionary<string, PolicyNet>();
        var driver = new RunDriver(pump, vocab);
        driver.Trace = null;
        string? line;
        while ((line = Console.In.ReadLine()) != null)
        {
            if (line.Length == 0) continue;
            var sw = Stopwatch.StartNew();
            var resp = new SearchResponse { status = "error" };
            Searcher? searcher = null;
            try
            {
                var req = JsonSerializer.Deserialize<SearchRequest>(line) ?? throw new InvalidDataException("empty request");
                resp.id = req.id;
                if (!nets.TryGetValue(req.model, out var net)) nets[req.model] = net = new PolicyNet(req.model);
                searcher = new Searcher(net) { K = req.k, N = req.n, MinP = req.minp };
                ulong want = ulong.Parse(req.hash);
                var save = RoomState.Deserialize(Convert.FromBase64String(req.save));
                int i = 0;
                bool done = false;
                driver.Policy = (d, enc) =>
                {
                    if (done) return 0;
                    if (i < req.decisions.Length)
                    {
                        int a = req.decisions[i++];
                        if (a < d.Actions.Count) return a;
                        resp.status = "desync"; resp.msg = $"decision {i - 1}: index {a} of {d.Actions.Count}";
                        done = true; driver.AbortRequested = true; return 0;
                    }
                    if (enc.B.Hash() != want)
                    {
                        resp.status = "desync"; resp.msg = $"hash after {i} decisions";
                        done = true; driver.AbortRequested = true; return 0;
                    }
                    var probs = net.Forward(enc.B).Probs;
                    var (act, cands, scores) = searcher.Search(driver, probs, (uint)HashCode.Combine(req.id, i));
                    if (searcher.IsChild) return act; // rollout continues in the child
                    resp.status = "ok"; resp.action = act; resp.greedy = Argmax(probs);
                    resp.cands = cands.ToArray(); resp.scores = scores;
                    done = true; driver.AbortRequested = true; return act;
                };
                driver.StartFromSave(save);
                driver.PlayToEnd();
                if (searcher.IsChild) searcher.ExitChild(driver);
                if (!done) { resp.status = "desync"; resp.msg = $"no decision after {i} decisions (error: {driver.Error?.Split('\n')[0] ?? "-"})"; }
            }
            catch (Exception e)
            {
                if (searcher?.IsChild == true) Searcher.ChildExit(0);
                resp.status = "error";
                resp.msg = e.GetType().Name + ": " + e.Message;
            }
            finally
            {
                if (searcher?.IsChild != true)
                    try { driver.EndRun(); } catch { }
            }
            resp.ms = sw.Elapsed.TotalMilliseconds;
            stdout.WriteLine(JsonSerializer.Serialize(resp));
            stdout.Flush();
        }
    }
}

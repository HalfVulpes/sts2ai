using System.Text.Json;
using Sts2Ai.Shared.Encoding;

namespace Sts2Sim.Engine;

/// <summary>One human/LLM-controlled run over newline JSON stdin/stdout. Never auto-chooses an action.</summary>
public static class TextSession
{
    public static void Run(Func<string, string, string> arg, PumpContext pump, Vocab vocab, Action<string> print)
    {
        int character = RunDriver.CharacterIndex(arg("--character", "Ironclad"));
        int ascension = int.Parse(arg("--ascension", "10"));
        ulong seed = ulong.Parse(arg("--runseed", "0"));
        if (character < 0 || ascension < 0 || ascension > 10) throw new ArgumentException("invalid run configuration");
        int decision = 0, invalid = 0;
        string prefixFile = arg("--resume-prefix", "");
        var prefix = prefixFile.Length == 0 ? Array.Empty<int>()
            : JsonSerializer.Deserialize<int[]>(File.ReadAllText(prefixFile))!;
        var driver = new RunDriver(pump, vocab);
        driver.Policy = (d, enc) =>
        {
            int id = decision++;
            if (id < prefix.Length)
            {
                int recorded = prefix[id];
                if (recorded < 0 || recorded >= d.Actions.Count)
                    throw new InvalidDataException($"resume prefix diverged at decision {id}");
                return recorded;
            }
            print(JsonSerializer.Serialize(new { type = "decision", decision_id = id,
                action_count = d.Actions.Count, decision_type = d.Type.ToString(), floor = driver.State.TotalFloor,
                act = driver.State.CurrentActIndex + 1, hp = driver.Player.Creature.CurrentHp,
                text = LlmPolicy.RenderPlayerView(driver, d) }));
            while (true)
            {
                string line = Console.ReadLine() ?? throw new EndOfStreamException("text player disconnected");
                try
                {
                    using var doc = JsonDocument.Parse(line);
                    var root = doc.RootElement;
                    int requested = root.GetProperty("decision_id").GetInt32();
                    int action = root.GetProperty("action").GetInt32();
                    if (requested != id || action < 0 || action >= d.Actions.Count)
                        throw new ArgumentException("stale decision or illegal action index");
                    return action;
                }
                catch (Exception e) when (e is JsonException or ArgumentException or KeyNotFoundException or InvalidOperationException)
                {
                    invalid++;
                    print(JsonSerializer.Serialize(new { type = "input_error", decision_id = id, error = e.Message }));
                }
            }
        };
        try
        {
            driver.StartRun(character, ascension, seed);
            driver.PlayToEnd();
            print(JsonSerializer.Serialize(new { type = "terminal", won = driver.Won, floor = driver.State.TotalFloor,
                act = driver.State.CurrentActIndex + 1, hp = driver.Player.Creature.CurrentHp, steps = driver.Steps,
                decisions = decision, invalid_inputs = invalid, error = driver.Error,
                cause = driver.Won ? "won" : driver.Stalemate ? "stalemate" : driver.NoOpLoop ? "noop"
                    : driver.Error != null ? "error" : "died", game_seed = seed.ToString(),
                seed_string = driver.SeedString, seed_protocol = driver.SeedProtocol }));
        }
        finally { if (driver.State != null) driver.EndRun(); }
    }
}

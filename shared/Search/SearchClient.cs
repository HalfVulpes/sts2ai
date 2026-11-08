using System.Diagnostics;
using System.Text.Json;
using MegaCrit.Sts2.Core.Multiplayer.Replay;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace Sts2Ai.Shared.Search;

/// <summary>
/// The room-entry state of the current room, as the game's CombatReplayWriter records it (enabled whenever
/// TestMode is off, i.e. in the live game): RecordInitialState(ToSave(null)) runs at room entry, before the
/// room's random setup, so restoring it and replaying the room's decisions reproduces the current decision.
/// </summary>
public static class RoomState
{
    static readonly System.Reflection.FieldInfo ReplayField = typeof(CombatReplayWriter)
        .GetField("_replay", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
        ?? throw new MissingFieldException("CombatReplayWriter._replay");

    /// <summary>The replay object of the current room: a new instance at every room entry (null before the first).</summary>
    public static CombatReplay? Current() => (CombatReplay?)ReplayField.GetValue(RunManager.Instance.CombatReplayWriter);

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
}

public sealed class SearchRequest
{
    public int id { get; set; }
    public string save { get; set; } = "";     // base64 of the room-entry SerializableRun
    public int[] decisions { get; set; } = Array.Empty<int>(); // decision indices taken in this room so far
    public string model { get; set; } = "";    // model file the client plays with
    public string hash { get; set; } = "";     // ObsBuilder.Hash() of the current decision (sync check)
    public int k { get; set; } = 3;
    public int n { get; set; } = 16;
    public float minp { get; set; } = 0.02f;
}

public sealed class SearchResponse
{
    public int id { get; set; }
    public string status { get; set; } = "";   // ok | desync | error
    public int action { get; set; } = -1;
    public int greedy { get; set; } = -1;
    public int[] cands { get; set; } = Array.Empty<int>();
    public double[] scores { get; set; } = Array.Empty<double>();
    public double ms { get; set; }
    public string msg { get; set; } = "";
}

/// <summary>
/// Client of `Sts2Sim.dll searchserver`: a persistent headless simulator that restores the room from its
/// room-entry state, replays the room's decisions, checks the observation hash and searches (fork-based). One JSON
/// request per line on stdin, one JSON response per line on stdout. Blocking: call from a background thread.
/// </summary>
public sealed class SearchClient : IDisposable
{
    readonly Process _proc;
    int _nextId;
    readonly object _lock = new();

    public SearchClient(string dotnet, string simDll)
    {
        var psi = new ProcessStartInfo(dotnet)
        {
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = false,
            UseShellExecute = false,
            WorkingDirectory = Path.GetDirectoryName(simDll)!,
        };
        psi.ArgumentList.Add(simDll);
        psi.ArgumentList.Add("searchserver");
        psi.ArgumentList.Add("--quiet");
        // fork safety (see Searcher in the simulator)
        psi.Environment["DOTNET_EnableWriteXorExecute"] = "0";
        psi.Environment["DOTNET_TieredCompilation"] = "0";
        psi.Environment["DOTNET_gcConcurrent"] = "0";
        _proc = Process.Start(psi) ?? throw new InvalidOperationException("could not start the search server");
    }

    public bool IsAlive => !_proc.HasExited;

    /// <summary>Send one request and wait for its response (null on timeout or if the server died).</summary>
    public SearchResponse? Search(SearchRequest req, TimeSpan timeout)
    {
        lock (_lock)
        {
            if (_proc.HasExited) return null;
            req.id = ++_nextId;
            _proc.StandardInput.WriteLine(JsonSerializer.Serialize(req));
            _proc.StandardInput.Flush();
            var deadline = DateTime.UtcNow + timeout;
            while (DateTime.UtcNow < deadline)
            {
                var task = _proc.StandardOutput.ReadLineAsync();
                if (!task.Wait(deadline - DateTime.UtcNow)) { Restartable = true; return null; }
                var line = task.Result;
                if (line == null) return null;
                if (!line.StartsWith("{\"id\"")) continue; // boot messages
                var resp = JsonSerializer.Deserialize<SearchResponse>(line);
                if (resp != null && resp.id == req.id) return resp;
            }
            Restartable = true;
            return null;
        }
    }

    /// <summary>Set after a timeout: the stream may hold a late response, so the client should be replaced.</summary>
    public bool Restartable { get; private set; }

    public void Dispose()
    {
        try { if (!_proc.HasExited) _proc.Kill(entireProcessTree: true); } catch { }
        _proc.Dispose();
    }
}

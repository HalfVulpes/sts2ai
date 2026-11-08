using Godot;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.TestSupport;
using Sts2Ai.Shared.Decisions;
using Sts2Ai.Shared.Encoding;
using Sts2Ai.Shared.Inference;
using Sts2Ai.Shared.Search;

namespace Sts2AiMod;

/// <summary>
/// Watches the live game, evaluates every decision with the policy and shows it in the overlay.
/// With autoplay on it carries out the policy's choice (one action per StepDelay so it can be
/// followed), and card selection prompts are answered by the policy through the game's
/// ICardSelector hook. Keys: F8 autoplay on/off, F9 play one step, F10 hide/show overlay; the overlay has the same
/// as buttons (keys do not reach the game in some remote-desktop sessions).
/// </summary>
static class RunDriverLimits
{
    public const int MaxCombatTurns = 150;    // keep equal to Sts2Sim.Engine.RunDriver.MaxCombatTurns
    public const int MaxActionsPerTurn = 300; // keep equal to Sts2Sim.Engine.RunDriver.MaxActionsPerTurn
}

public sealed class AiController : ICardSelector
{
    readonly string _modelDir;
    readonly Dictionary<string, PolicyNet> _nets = new();
    PolicyNet _net = null!;
    readonly DecisionBuilder _build;
    readonly Pilot _pilot;

    public bool Autoplay { get; private set; }
    public bool Visible { get; private set; } = true;
    public float StepDelay = 0.8f;      // seconds between autoplayed actions, so a human can follow
    /// <summary>Seconds between looks at the game state. Detecting and encoding allocates; doing it every frame
    /// made the garbage collector run constantly (suspected trigger of native crashes in the game).</summary>
    public float DetectInterval = 0.1f;
    double _sinceDetect = 999;

    public Decision? Current { get; private set; }
    public PolicyNet.Output? Output { get; private set; }
    public string Context { get; private set; } = "waiting for a run";

    IDisposable? _selectorScope;
    int _turnSeen = -1, _turnActions;
    long _execSig;                       // decision signature when we last acted
    string _execLabel = "";
    readonly HashSet<string> _blocked = new(); // actions that left the decision unchanged (no-ops)
    Task? _busy;
    long _lastSig;
    double _sinceAct = 999;
    bool _stepRequested;
    readonly Dictionary<Key, bool> _keyDown = new();

    public AiController(string modelPath, string vocabPath)
    {
        _modelDir = Path.GetDirectoryName(modelPath)!;
        _net = _nets["default"] = new PolicyNet(modelPath);
        var vocab = Vocab.Load(vocabPath);
        _build = new DecisionBuilder(new StateEncoder(vocab), vocab, new RunHistorySummary());
        _pilot = new Pilot(_build);
        LoadSearchConfig();
        AppDomain.CurrentDomain.ProcessExit += (_, _) => _client?.Dispose();
    }

    // ------------------------------------------------------------------ search (search.cfg + Sts2Sim searchserver)
    // At uncertain decisions in boss/elite fights the mod asks a headless simulator (the search server, started
    // on first use) to search: it restores the room from the room-entry state the game records (RoomState),
    // replays the decisions the AI took in this room (_roomLog), checks that it sees the same observation (hash)
    // and evaluates the top actions with forked rollouts. Any mismatch (e.g. the player acted by hand in this
    // room), timeout or error falls back to the network's own choice.

    sealed class SearchConfig
    {
        public string dotnet { get; set; } = "";
        public string sim { get; set; } = "";
        public int k { get; set; } = 3;
        public int n { get; set; } = 16;
        public float conf { get; set; } = 0.95f;
        public float minp { get; set; } = 0.02f;
        public string rooms { get; set; } = "elite";   // boss | elite (= boss and elite) | all
        public double timeout { get; set; } = 60;
    }

    SearchConfig? _searchCfg;
    SearchClient? _client;
    Task<SearchResponse?>? _searchTask;
    long _searchSig;
    DateTime _searchStart;
    object? _roomReplay;
    readonly List<int> _roomLog = new();
    bool _roomDesync;
    public bool SearchEnabled { get; private set; }
    public bool SearchAvailable => _searchCfg != null;
    public string SearchStatus { get; private set; } = "";

    void LoadSearchConfig()
    {
        // search.cfg (JSON; not .json: the game reads every .json in a mod folder as a manifest)
        var path = Path.Combine(_modelDir, "search.cfg");
        try
        {
            if (!File.Exists(path)) return;
            var cfg = System.Text.Json.JsonSerializer.Deserialize<SearchConfig>(File.ReadAllText(path));
            if (cfg == null || !File.Exists(cfg.sim) || !File.Exists(cfg.dotnet))
            {
                SearchStatus = "search.cfg: simulator or dotnet not found";
                return;
            }
            _searchCfg = cfg;
            SearchEnabled = true;
            Log.Info($"[sts2ai] search available ({cfg.sim}, k={cfg.k} n={cfg.n} rooms={cfg.rooms})");
        }
        catch (Exception e) { SearchStatus = "search.cfg: " + e.Message; }
    }

    public void ToggleSearch() { if (SearchAvailable) SearchEnabled = !SearchEnabled; }

    /// <summary>A new room-entry record means a new room: start a new decision log.</summary>
    void TrackRoom()
    {
        var rep = RoomState.Current();
        if (!ReferenceEquals(rep, _roomReplay)) { _roomReplay = rep; _roomLog.Clear(); _roomDesync = false; }
    }

    /// <summary>Log a decision as the simulator's driver sees it (it skips forced non-combat decisions).</summary>
    void LogDecision(Decision d, int index)
    {
        if (d.Actions.Count > 1 || d.Type == DecisionType.Combat) _roomLog.Add(index);
    }

    bool WantSearch(Decision d, float[] probs)
    {
        var cfg = _searchCfg;
        if (cfg == null || !SearchEnabled || _roomDesync || d.Type != DecisionType.Combat || d.Actions.Count < 2) return false;
        if (RunManager.Instance.NetService is not { Type: MegaCrit.Sts2.Core.Multiplayer.Game.NetGameType.Singleplayer }) return false;
        if (RoomState.Current() == null) return false;
        var room = RunManager.Instance.DebugOnlyGetState()?.CurrentRoom as MegaCrit.Sts2.Core.Rooms.CombatRoom;
        if (room == null) return false;
        bool kind = cfg.rooms == "all"
                    || cfg.rooms == "boss" && room.RoomType == MegaCrit.Sts2.Core.Rooms.RoomType.Boss
                    || cfg.rooms == "elite" && room.RoomType is MegaCrit.Sts2.Core.Rooms.RoomType.Boss or MegaCrit.Sts2.Core.Rooms.RoomType.Elite;
        return kind && probs.Max() < cfg.conf;
    }

    void StartSearch(long sig)
    {
        var cfg = _searchCfg!;
        try
        {
            if (_client is { IsAlive: false } or { Restartable: true }) { _client.Dispose(); _client = null; }
            _client ??= new SearchClient(cfg.dotnet, cfg.sim);
        }
        catch (Exception e)
        {
            SearchStatus = "search server failed to start: " + e.Message;
            SearchEnabled = false;
            return;
        }
        var req = new SearchRequest
        {
            save = Convert.ToBase64String(RoomState.Serialize(RoomState.Current()!.serializableRun)),
            decisions = _roomLog.ToArray(),
            model = Path.Combine(_modelDir, ModelName),
            hash = _build.Enc.B.Hash().ToString(),
            k = cfg.k, n = cfg.n, minp = cfg.minp,
        };
        var client = _client;
        _searchSig = sig;
        _searchStart = DateTime.UtcNow;
        _searchTask = Task.Run(() => client.Search(req, TimeSpan.FromSeconds(cfg.timeout)));
    }

    /// <summary>The searched action for this decision, or -1 (fall back to the network's choice).</summary>
    int SearchResult(Decision d, SearchResponse? r)
    {
        if (r == null) { SearchStatus = "search: timeout (normal play)"; return -1; }
        if (r.status == "desync")
        {
            _roomDesync = true; // e.g. the player acted by hand in this room: no search until the next room
            SearchStatus = $"search: not in sync ({r.msg}); normal play this room";
            Log.Info("[sts2ai] search desync: " + r.msg);
            return -1;
        }
        if (r.status != "ok" || r.action < 0 || r.action >= d.Actions.Count)
        {
            SearchStatus = $"search: {r.status} {r.msg}";
            return -1;
        }
        var parts = r.cands.Select((c, i) => $"{(c < d.Actions.Count ? d.Actions[c].Label : "?")} {r.scores[i]:F2}");
        SearchStatus = $"search {r.ms / 1000:F1}s: " + string.Join(" | ", parts) + (r.action != r.greedy ? "  (changed)" : "");
        return r.action;
    }

    bool Pressed(Key k)
    {
        bool down = Input.IsKeyPressed(k);
        bool was = _keyDown.GetValueOrDefault(k);
        _keyDown[k] = down;
        return down && !was;
    }

    // overlay buttons: same as F8 / F9 / F10
    public void ToggleAutoplay() => SetAutoplay(!Autoplay);
    public void RequestStep() => _stepRequested = true;
    public void ToggleVisible() => Visible = !Visible;

    void SetAutoplay(bool on)
    {
        if (on == Autoplay) return;
        Autoplay = on;
        UpdateSelector();
        Log.Info($"[sts2ai] autoplay {(on ? "on" : "off")}");
    }

    /// <summary>
    /// The card-selection hook (ICardSelector) answers prompts directly, skipping the multiplayer choice
    /// synchronization (ReserveChoiceId / SyncLocalChoice): in co-op the host never learns the choice and the
    /// run desyncs. So it is used only in single-player runs; in multiplayer the game's own selection screens
    /// appear (synchronized) and are answered through the UI.
    /// </summary>
    void UpdateSelector()
    {
        bool singleplayer = RunManager.Instance.NetService is { Type: MegaCrit.Sts2.Core.Multiplayer.Game.NetGameType.Singleplayer };
        bool want = Autoplay && singleplayer && RunManager.Instance.DebugOnlyGetState() != null;
        if (want && _selectorScope == null) _selectorScope = CardSelectCmd.PushSelector(this);
        else if (!want && _selectorScope != null) { _selectorScope.Dispose(); _selectorScope = null; }
    }

    public void Tick(double dt)
    {
        if (Pressed(Key.F8)) ToggleAutoplay();
        if (Pressed(Key.F9)) RequestStep();
        if (Pressed(Key.F10)) ToggleVisible();
        _sinceAct += dt;
        _sinceDetect += dt;
        if (!_stepRequested && _sinceDetect < DetectInterval) return;
        _sinceDetect = 0;

        var state = RunManager.Instance.DebugOnlyGetState();
        var player = state == null ? null : LocalContext.GetMe(state.Players);
        if (state == null || player == null)
        {
            Clear("waiting for a run");
            return;
        }
        SelectModel(player);
        UpdateSelector();
        if (_busy != null)
        {
            // an action whose game task waits on a screen the AI must answer would block forever: after 2 s,
            // stop waiting and look at the game again (a later fault is still logged)
            if (!_busy.IsCompleted && _sinceAct < 2.0) return;
            if (_busy.IsFaulted) Log.Error("[sts2ai] action failed: " + _busy.Exception);
            else if (!_busy.IsCompleted)
            {
                _busy.ContinueWith(x => Log.Error("[sts2ai] action failed: " + x.Exception), TaskContinuationOptions.OnlyOnFaulted);
                Log.Info("[sts2ai] action still running after 2 s; continuing");
            }
            _busy = null;
        }

        if (player.PlayerCombatState is { TurnNumber: > RunDriverLimits.MaxCombatTurns } && Autoplay)
        {
            SetAutoplay(false); // same limit as training: a fight this long is a stalemate, hand it to the player
            Context = "stalemate: autoplay paused";
        }
        var pending = _pilot.Detect(state, player, out var waitingFor);
        if (pending == null)
        {
            _stepRequested = false;
            if (waitingFor == "run over") SetAutoplay(false);
            Context = waitingFor.Length > 0 ? waitingFor : "watching";
            return;
        }

        Evaluate(pending.Decision, pending.Context);
        bool go = _stepRequested || (Autoplay && _sinceAct >= StepDelay);
        if (!go) return;
        _stepRequested = false;
        var d = pending.Decision;
        long sig = Signature(_build.Enc.B);
        // the previous action changed nothing (e.g. the game refused it): don't repeat it here
        if (sig == _execSig && _execLabel.Length > 0) _blocked.Add(_execLabel);
        else if (sig != _execSig) _blocked.Clear();
        TrackRoom();
        int index = d.Actions.Count == 0 ? -1 : BestAllowed(d, Output!.Probs);
        if (index >= 0 && WantSearch(d, Output!.Probs))
        {
            if (_searchTask == null || _searchSig != sig)
            {
                StartSearch(sig);
                if (_searchTask != null) { Context = "searching..."; _stepRequested = _stepRequested || !Autoplay; return; }
            }
            else if (!_searchTask.IsCompleted)
            {
                Context = $"searching... {(DateTime.UtcNow - _searchStart).TotalSeconds:F0}s";
                _stepRequested = _stepRequested || !Autoplay;
                return;
            }
            else
            {
                int s = SearchResult(d, _searchTask.Result);
                _searchTask = null;
                if (s >= 0 && !_blocked.Contains(d.Actions[s].Label)) index = s;
            }
        }
        var action = index < 0 ? new LegalAction() : d.Actions[index];
        if (index >= 0) LogDecision(d, index);
        _execSig = sig;
        _execLabel = action.Label;
        if (d.Type == DecisionType.Combat && player.PlayerCombatState is { } pcs)
        {
            if (pcs.TurnNumber != _turnSeen) { _turnSeen = pcs.TurnNumber; _turnActions = 0; }
            if (++_turnActions > RunDriverLimits.MaxActionsPerTurn)
            {
                SetAutoplay(false); // same limit as training: an endless turn is a stalemate, hand it to the player
                Context = "endless turn: autoplay paused";
                return;
            }
        }
        Log.Info($"[sts2ai] {pending.Context}: {action}");
        _busy = ExecuteDeferred(pending, action);
        _sinceAct = 0;
        _lastSig = 0; // re-evaluate after the game reacted
    }

    /// <summary>
    /// Run the action at the end of the frame (deferred call), not inside the SceneTree.ProcessFrame signal we
    /// are ticked from: game actions create and free nodes, which Godot wants done outside signal processing.
    /// </summary>
    static Task ExecuteDeferred(Pilot.Pending pending, LegalAction action)
    {
        var done = new TaskCompletionSource();
        async Task Run()
        {
            try { await pending.Execute(action); done.TrySetResult(); }
            catch (Exception e) { done.TrySetException(e); }
        }
        // a void Action (an async lambda would be a Func<Task>, whose result Godot cannot convert to a Variant)
        Callable.From(() => { _ = Run(); }).CallDeferred();
        return done.Task;
    }

    /// <summary>Use model_&lt;character&gt;.bin (a specialist) when shipped, otherwise model.bin.</summary>
    /// <summary>Runs at this ascension or above use model_&lt;character&gt;_hi.bin when present (trained at A8-A10),
    /// so a model specialised for high ascension never replaces the one for lower levels.</summary>
    public const int HiAscension = 8;
    public string ModelName { get; private set; } = "model.bin";

    PolicyNet Net(string slot, Func<PolicyNet> fallback)
    {
        if (_nets.TryGetValue(slot, out var net)) return net;
        var path = Path.Combine(_modelDir, $"model_{slot}.bin");
        if (File.Exists(path)) { net = new PolicyNet(path); Log.Info($"[sts2ai] loaded {path}"); }
        else net = fallback();
        return _nets[slot] = net;
    }

    void SelectModel(MegaCrit.Sts2.Core.Entities.Players.Player player)
    {
        string key = player.Character.Id.Entry.ToLowerInvariant();
        bool hi = player.RunState.AscensionLevel >= HiAscension;
        var baseNet = Net(key, () => _nets["default"]);
        var net = hi ? Net(key + "_hi", () => baseNet) : baseNet;
        ModelName = net == _nets["default"] ? "model.bin" : net == baseNet ? $"model_{key}.bin" : $"model_{key}_hi.bin";
        if (net != _net) { _net = net; _lastSig = 0; }
        // a co-op-trained model sees teammates as ally tokens; a solo-trained one must not (unseen inputs)
        StateEncoder.EncodeAllies = _net.EncodeAllies;
    }

    /// <summary>Run the network only when the encoded decision changed.</summary>
    void Evaluate(Decision d, string context)
    {
        Context = context;
        long sig = Signature(_build.Enc.B);
        if (sig == _lastSig && Current != null) return;
        _lastSig = sig;
        Current = d;
        Output = _net.Forward(_build.Enc.B);
    }

    void Clear(string context)
    {
        Current = null;
        Output = null;
        Context = context;
        _lastSig = 0;
    }

    static long Signature(ObsBuilder b)
    {
        var h = new HashCode();
        h.Add(b.NumTokens);
        h.Add(b.NumActions);
        for (int i = 0; i < b.NumTokens * ObsSpec.NTokCat; i++) h.Add(b.TokCat[i]);
        for (int i = 0; i < b.NumTokens * ObsSpec.FTok; i++) h.Add(b.TokNum[i]);
        for (int i = 0; i < b.NumActions * ObsSpec.NActCat; i++) h.Add(b.ActCat[i]);
        return (uint)h.ToHashCode() | 1L << 40;
    }

    int BestAllowed(Decision d, float[] p)
    {
        int best = -1;
        for (int i = 0; i < p.Length && i < d.Actions.Count; i++)
            if (!_blocked.Contains(d.Actions[i].Label) && (best < 0 || p[i] > p[best])) best = i;
        return best < 0 ? ArgMax(p) : best;
    }

    static int ArgMax(float[] p)
    {
        int best = 0;
        for (int i = 1; i < p.Length; i++) if (p[i] > p[best]) best = i;
        return best;
    }

    // ------------------------------------------------------------------ ICardSelector (autoplay only)

    LegalAction Choose(Decision d)
    {
        var o = _net.Forward(_build.Enc.B);
        Current = d;
        Output = o;
        TrackRoom();
        int i = ArgMax(o.Probs);
        LogDecision(d, i);
        return d.Actions[i];
    }

    public Task<IEnumerable<CardModel>> GetSelectedCards(IEnumerable<CardModel> options, int minSelect, int maxSelect)
    {
        var state = RunManager.Instance.DebugOnlyGetState()!;
        var player = LocalContext.GetMe(state.Players)!;
        var opts = options.ToList();
        var picked = new List<CardModel>();
        maxSelect = Math.Min(maxSelect, opts.Count);
        minSelect = Math.Min(minSelect, maxSelect);
        while (picked.Count < maxSelect)
        {
            var d = _build.CardSelect(state, player, opts, picked, minSelect, maxSelect);
            var a = Choose(d);
            Log.Info("[sts2ai] card select: " + a);
            if (a.Type == ActionType.Confirm) break;
            picked.Add((CardModel)a.Payload!);
        }
        return Task.FromResult<IEnumerable<CardModel>>(picked);
    }

    public CardRewardSelection GetSelectedCardReward(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives)
    {
        var state = RunManager.Instance.DebugOnlyGetState()!;
        var player = LocalContext.GetMe(state.Players)!;
        var a = Choose(_build.CardReward(state, player, options.Select(o => o.Card).ToList(), alternatives));
        _build.NoteCardRewardChoice(a);
        return a.Type == ActionType.SelectCard
            ? new CardRewardSelection { card = (CardModel)a.Payload! }
            : new CardRewardSelection { alternative = a.Payload as CardRewardAlternative };
    }
}

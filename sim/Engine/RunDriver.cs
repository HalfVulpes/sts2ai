using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.TestSupport;
using MegaCrit.Sts2.Core.Unlocks;
using Sts2Ai.Shared.Decisions;
using Sts2Ai.Shared.Encoding;

namespace Sts2Sim.Engine;

/// <summary>
/// Plays one full run of the real game logic, asking <see cref="Policy"/> at every decision.
/// Top-level decisions (combat play phase, map, events, rest site, shop, treasure) are detected
/// from game state between pump cycles. Decisions requested from *inside* game logic (card
/// rewards, card selection prompts, rewards screens) arrive through the test hooks
/// (ICardSelector, RewardsSet.testSelector) and call the policy synchronously.
/// </summary>
public sealed partial class RunDriver : ICardSelector
{
    public delegate int PolicyFn(Decision d, StateEncoder enc);

    readonly PumpContext _pump;
    readonly Vocab _vocab;
    public readonly StateEncoder Enc;
    readonly RunHistorySummary _history = new();
    readonly DecisionBuilder _build;
    public PolicyFn Policy = (_, _) => 0;
    public Action<string>? Trace;

    public RunState State { get; private set; } = null!;
    public Player Player { get; private set; } = null!;
    public bool Over { get; private set; }
    public bool Won { get; private set; }
    public string? Error { get; private set; }
    public int Steps { get; private set; }
    public int CombatId { get; private set; }
    public int CombatsWon { get; private set; }
    public int BossKills { get; private set; }
    public int EliteKills { get; private set; }
    public int MaxSteps = 20000;
    /// <summary>Testing only: huge HP so random play reaches the end of the game.</summary>
    public bool GodMode;
    public static bool CanonicalSeeds;
    public string SeedString { get; private set; } = "";
    public string SeedProtocol { get; private set; } = "legacy-numeric";
    /// <summary>Set (e.g. from the policy) to stop the run at the next top-level step; the run is left as is.</summary>
    public bool AbortRequested;

    // per-room bookkeeping
    AbstractRoom? _room;
    bool _roomDone;
    bool _treasureOpened;
    bool _pendingNextAct;
    bool _roomPrefinishedAtEntry;
    bool _shopChecked;
    bool _progressed;
    MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame? _sphere;
    readonly List<Task> _relicObtains = new();
    readonly Queue<string> _recent = new();
    string _lastKey = "";
    int _repeat;
    Task? _inFlight;
    /// <summary>First exception the game logged during this run (game code swallows/logs them via TaskHelper).</summary>
    string? _gameException;

    static readonly string[] CharacterNames = { "Ironclad", "Silent", "Defect", "Necrobinder", "Regent" };
    public static CharacterModel CharacterByIndex(int i) =>
        ModelDb.AllCharacters.First(c => c.Id.Entry.Equals(CharacterNames[i], StringComparison.OrdinalIgnoreCase));
    public static int CharacterIndex(string name) => Array.FindIndex(CharacterNames, n => n.Equals(name, StringComparison.OrdinalIgnoreCase));

    public RunDriver(PumpContext pump, Vocab vocab)
    {
        _pump = pump;
        _vocab = vocab;
        Enc = new StateEncoder(vocab);
        _build = new DecisionBuilder(Enc, vocab, _history);
        CardSelectCmd.UseSelector(this);
        RewardsSet.testSelector = OfferRewards;
        GamePatches.SphereCaptured = g => _sphere = g;
        GamePatches.BundleChooser = ChooseBundle;
        MegaCrit.Sts2.Core.Logging.Log.LogCallback += (level, text, _) =>
        {
            if (level < MegaCrit.Sts2.Core.Logging.LogLevel.Error || !text.Contains("Exception") || State == null) return;
            // exception type + message and the throwing frame
            _gameException ??= string.Join(" ", text.Split('\n').Take(2).Select(l => l.Trim()));
        };
        CombatManager.Instance.CombatWon += room =>
        {
            CombatsWon++;
            if (room.RoomType == RoomType.Boss) BossKills++;
            if (room.RoomType == RoomType.Elite) EliteKills++;
        };
    }

    // ================================================================== run lifecycle

    public void StartRun(int character, int ascension, ulong seed) => StartRun(new[] { character }, ascension, seed);

    public void StartRun(int character, int ascension, string gameSeed)
    {
        string canonical = MegaCrit.Sts2.Core.Helpers.SeedHelper.CanonicalizeSeed(gameSeed);
        StartRunCore(new[] { character }, ascension, canonical,
            new MegaCrit.Sts2.Core.Random.Rng((uint)MegaCrit.Sts2.Core.Helpers.StringHelper.GetDeterministicHashCode(canonical), "act_selection"),
            "canonical-game");
    }

    /// <summary>Start a run; more than one character makes it a co-op run (see Coop, RunDriver.Coop.cs).</summary>
    public void StartRun(int[] characters, int ascension, ulong seed)
    {
        if (CanonicalSeeds)
        {
            string canonical = CanonicalSeedFromNumber(seed);
            StartRunCore(characters, ascension, canonical,
                new MegaCrit.Sts2.Core.Random.Rng((uint)MegaCrit.Sts2.Core.Helpers.StringHelper.GetDeterministicHashCode(canonical), "act_selection"),
                "canonical-game");
        }
        else
            StartRunCore(characters, ascension, SeedFromNumber(seed),
                new MegaCrit.Sts2.Core.Random.Rng((uint)(seed ^ (seed >> 32))), "legacy-numeric");
    }

    void StartRunCore(int[] characters, int ascension, string seedStr, MegaCrit.Sts2.Core.Random.Rng rng, string protocol)
    {
        if (State != null) EndRun();
        Over = Won = Stalemate = NoOpLoop = AbortRequested = false;
        Error = null;
        Steps = CombatId = CombatsWon = BossKills = EliteKills = 0;
        _room = null;
        _roomDone = _treasureOpened = _pendingNextAct = false;
        _inFlight = null;
        _gameException = null;
        _recent.Clear();
        _lastKey = "";
        _repeat = 0;
        _turnSeen = -1;
        _turnActions = 0;
        SeedString = seedStr;
        SeedProtocol = protocol;
        var players = characters.Select((c, i) => Player.CreateForNewRun(CharacterByIndex(c), UnlockState.all, (ulong)(i + 1))).ToArray();
        var player = players[0];
        var acts = ActModel.GetRandomList(rng, UnlockState.all, false).Select(a => a.ToMutable()).ToList();
        State = RunState.CreateForNewRun(players, acts, Array.Empty<ModifierModel>(), GameMode.Standard, ascension, seedStr);
        Player = player;
        Players = players;
        Coop.Active = players.Length > 1;
        Coop.Players = players;
        Coop.Pending.Clear();
        if (!Coop.Active)
            RunManager.Instance.SetUpNewSingleplayer(State, false);
        else
        {
            // host with all players in-process; SetUpTest is SetUpNew* minus room generation
            RunManager.Instance.SetUpTest(State, new Coop.LoopbackHostNetService(1), disableCombatStateSync: true, shouldSave: false);
            RunManager.Instance.GenerateRooms();
            ResetCoopRun();
        }
        RunManager.Instance.TreasureRoomRelicSynchronizer.RelicsAwarded += OnRelicsAwarded;
        _sphere = null;
        Await(RunManager.Instance.FinalizeStartingRelics());
        if (GodMode) foreach (var p in players) { p.Creature.SetMaxHpInternal(99999); p.Creature.SetCurrentHpInternal(99999); }
        RunManager.Instance.Launch();
        _inFlight = RunManager.Instance.EnterAct(0, doTransition: false);
    }

    static string SeedFromNumber(ulong v)
    {
        const string alphabet = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ";
        var s = new char[10];
        for (int i = 0; i < s.Length; i++) { s[i] = alphabet[(int)(v % (ulong)alphabet.Length)]; v /= (ulong)alphabet.Length; }
        return new string(s);
    }

    public static string CanonicalSeedFromNumber(ulong v)
    {
        const string alphabet = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ";
        const ulong limit = 2064377754059776UL; // 34^10; conversion is injective below this bound.
        if (v >= limit) throw new ArgumentOutOfRangeException(nameof(v), "canonical numeric seed must be below 34^10");
        var s = new char[10];
        for (int i = 0; i < s.Length; i++) { s[i] = alphabet[(int)(v % 34)]; v /= 34; }
        return new string(s);
    }

    public void EndRun()
    {
        try
        {
            RunManager.Instance.CleanUp();
            Settle();
        }
        catch (Exception e) { Trace?.Invoke("cleanup error: " + e.Message); }
        State = null!;
    }

    /// <summary>Play the whole run. Returns when the run is won, lost, or errors out.</summary>
    public void PlayToEnd()
    {
        try
        {
            while (!Over && !AbortRequested)
            {
                Settle();
                if (_gameException != null) { Fail("game exception: " + _gameException); break; }
                CheckOver();
                if (Over) break;
                var d = DetectTopLevel();
                if (d == null)
                {
                    if (_progressed) { _progressed = false; continue; }
                    if (!WaitForProgress()) Fail("stuck: no decision available (" + Describe() + ")");
                    continue;
                }
                if (Coop.Active) ExecuteCoop(d, Ask(d));
                else Execute(d, Ask(d));
            }
        }
        catch (RunAbortedException) { }
        catch (Exception e) { Fail(e.ToString()); }
    }

    sealed class RunAbortedException : Exception { }

    void Fail(string why)
    {
        Error ??= why;
        Over = true;
        Trace?.Invoke("ERROR " + why);
    }

    /// <summary>Player turns after which a single combat counts as lost (a stalemate where the policy only blocks).</summary>
    public const int MaxCombatTurns = 150;
    /// <summary>Actions within a single player turn after which the combat counts as lost.</summary>
    public const int MaxActionsPerTurn = 300;
    int _turnSeen = -1, _turnActions;
    public bool Stalemate { get; private set; }
    public bool NoOpLoop { get; private set; }

    void CheckOver()
    {
        if (State == null) return;
        if (Coop.Active) { CheckOverCoop(); return; }
        if (CombatManager.Instance.IsInProgress && Player.PlayerCombatState is { TurnNumber: > MaxCombatTurns })
        {
            // the game has no turn limit, but a fight that never ends is a lost run; ending it as a loss
            // (not an error) makes the policy learn not to stall
            Stalemate = true;
            Over = true;
            Won = false;
            return;
        }
        if (Player.Creature.IsDead || State.IsGameOver)
        {
            Over = true;
            Won = Won || (State.CurrentRoom is EventRoom er && er.CanonicalEvent is MegaCrit.Sts2.Core.Models.Events.TheArchitect && !Player.Creature.IsDead) || RunVictory();
        }
    }

    bool RunVictory() => State.CurrentActIndex >= State.Acts.Count - 1 && State.CurrentRoom?.IsVictoryRoom == true;

    // ================================================================== pumping

    void Settle()
    {
        int ticks = 0;
        for (int i = 0; i < 100000; i++)
        {
            // pending game-side waits (PumpContext.Delay) advance only once everything else is idle;
            // bounded so a wait loop that never ends (e.g. executor left paused) cannot spin forever
            if (_pump.Drain() == 0 && (ticks++ >= 1000 || !_pump.ReleaseTicks())) return;
        }
    }

    void Await(Task t)
    {
        var deadline = DateTime.UtcNow.AddSeconds(30);
        while (!t.IsCompleted)
        {
            if (_pump.Drain() == 0 && !_pump.ReleaseTicks() && !_pump.WaitForWork(TimeSpan.FromMilliseconds(20)) && DateTime.UtcNow > deadline)
                throw new TimeoutException("await timeout");
        }
        t.GetAwaiter().GetResult();
    }

    /// <summary>Nothing to decide yet: wait briefly for timers (rare Task.Delay in game code).</summary>
    public readonly System.Diagnostics.Stopwatch WaitTime = new();
    public int Waits;
    bool WaitForProgress()
    {
        Waits++;
        WaitTime.Start();
        try { return WaitForProgressInner(); } finally { WaitTime.Stop(); }
    }

    bool WaitForProgressInner()
    {
        var deadline = DateTime.UtcNow.AddSeconds(5);
        while (DateTime.UtcNow < deadline)
        {
            // a finished driver task is the common case: check it before blocking on the queue
            if (_inFlight != null && _inFlight.IsCompleted)
            {
                var t = _inFlight;
                _inFlight = null;
                if (t.IsFaulted) Fail("task faulted: " + t.Exception);
                return true;
            }
            if (_pump.ReleaseTicks() || _pump.WaitForWork(TimeSpan.FromMilliseconds(10)))
            {
                Settle();
                return true;
            }
        }
        return false;
    }

    string Describe()
    {
        var pcs = Player.PlayerCombatState;
        var cm = CombatManager.Instance;
        var rm = RunManager.Instance;
        return $"room={State.CurrentRoom?.GetType().Name} combat={cm.IsInProgress} phase={pcs?.Phase} " +
               $"exec={rm.ActionExecutor.IsRunning} paused={rm.ActionExecutor.IsPaused} roomDone={_roomDone} " +
               $"overOrEnding={cm.IsOverOrEnding} disabled={cm.PlayerActionsDisabled} readyEnd={(pcs != null && cm.IsPlayerReadyToEndTurn(Player))} " +
               $"queueEmpty={rm.ActionQueueSet.IsEmpty} inFlight={(_inFlight == null ? "none" : _inFlight.Status.ToString())} " +
               $"syncState={rm.ActionQueueSynchronizer.CombatState} current={rm.ActionExecutor.CurrentlyRunningAction} recent: " + string.Join(" | ", _recent.TakeLast(6));
    }

    int Ask(Decision d)
    {
        if (++Steps > MaxSteps) { Fail("step limit; recent: " + string.Join(" | ", _recent)); throw new RunAbortedException(); }
        if (d.Actions.Count == 0) { Fail("decision without actions: " + d.Type); throw new RunAbortedException(); }
        if (d.Actions.Count == 1 && d.Type != DecisionType.Combat) return 0; // forced: don't bother the policy
        int a = Policy(d, Enc);
        if (a < 0 || a >= d.Actions.Count) a = 0;
        // the same non-combat choice over and over means it has no effect (a no-op the game refuses):
        // surface it as an error instead of burning the step budget
        if (d.Type == DecisionType.Combat && Player.PlayerCombatState is { } pcs)
        {
            // endless plays within one turn (a 0-cost card that keeps returning, ...) never advance the turn
            // counter, so the per-combat turn cap cannot catch them: cap the actions per turn as well
            if (pcs.TurnNumber != _turnSeen) { _turnSeen = pcs.TurnNumber; _turnActions = 0; }
            if (++_turnActions > MaxActionsPerTurn)
            {
                Trace?.Invoke("endless turn (counted as a loss)");
                Stalemate = true;
                Over = true;
                Won = false;
                throw new RunAbortedException();
            }
        }
        string key = d.Type + "|" + d.Actions[a];
        _repeat = key == _lastKey && d.Type != DecisionType.Combat ? _repeat + 1 : 0;
        _lastKey = key;
        if (_repeat >= 30)
        {
            // a loop that never ends the decision is a lost run, not an error: ending it as a loss means
            // the policy cannot profit from exploiting such loops (it gets no death penalty otherwise)
            Trace?.Invoke("no-op loop (counted as a loss): " + key);
            NoOpLoop = true;
            Over = true;
            Won = false;
            throw new RunAbortedException();
        }
        _recent.Enqueue($"[{d.Type}] {d.Actions[a]}");
        if (_recent.Count > 40) _recent.Dequeue();
        Trace?.Invoke($"{(Coop.Active ? $"P{Player.NetId} " : "")}[{d.Type}] -> {d.Actions[a]}  (of {d.Actions.Count})");
        return a;
    }

    // ================================================================== top-level decisions

    Decision? DetectTopLevel()
    {
        if (Coop.Active) return DetectTopLevelCoop();
        var room = State.CurrentRoom;
        if (!ReferenceEquals(room, _room))
        {
            _room = room;
            Trace?.Invoke($"ROOM act={State.CurrentActIndex + 1} floor={State.TotalFloor} {room?.RoomType} {(room as CombatRoom)?.Encounter.Id.Entry ?? (room as EventRoom)?.CanonicalEvent.Id.Entry}");
            _roomDone = false;
            _treasureOpened = false;
            _shopChecked = false;
            _roomPrefinishedAtEntry = room?.IsPreFinished == true && !CombatManager.Instance.IsInProgress && room is not CombatRoom { CombatState: not null } ? room.IsPreFinished : false;
        }
        if (_sphere != null)
        {
            if (!_sphere.IsFinished) { _inFlight = PlaySphere(_sphere); _sphere = null; return null; }
            _sphere = null;
        }
        if (CombatManager.Instance.IsInProgress)
        {
            return AwaitingCombatInput() ? CombatDecision() : null;
        }
        if (_inFlight != null && !_inFlight.IsCompleted) return null;
        if (room is EventRoom { IsVictoryRoom: true } vr)
        {
            // final dialogue event: advance until the game declares victory
            var opts = vr.LocalMutableEvent.CurrentOptions;
            int idx = -1;
            for (int i = 0; i < opts.Count; i++) if (!opts[i].IsLocked) { idx = i; break; }
            if (idx < 0) { Won = true; Over = true; return null; }
            RunManager.Instance.EventSynchronizer.ChooseLocalOption(idx);
            Settle();
            if (Player.Creature.IsDead || State.IsGameOver || RunManager.Instance.WinTime > 0 && opts.Count == 0) { Won = true; Over = true; }
            _progressed = true;
            return null;
        }
        switch (room)
        {
            case EventRoom er when !_roomDone:
            {
                var ev = er.LocalMutableEvent;
                if (!ev.IsFinished)
                {
                    var d = EventDecision(ev);
                    if (d != null) return d;
                }
                _roomDone = true;
                break;
            }
            case RestSiteRoom rs when !_roomDone:
                return RestDecision(rs);
            case MerchantRoom mr when !_roomDone:
                return ShopDecision(mr);
            case TreasureRoom tr when !_roomDone:
                return TreasureDecision(tr);
            case CombatRoom cr when !_roomDone:
                // combat won: replicate NCombatUi.OnCombatWon -> rewards screen -> proceed
                _roomDone = true;
                _inFlight = AfterCombat(cr);
                return null;
        }
        return MapDecision();
    }

    bool AwaitingCombatInput()
    {
        var cm = CombatManager.Instance;
        var rm = RunManager.Instance;
        var pcs = Player.PlayerCombatState;
        return cm.IsInProgress && !cm.IsOverOrEnding && pcs != null && pcs.Phase == PlayerTurnPhase.Play
               && !cm.PlayerActionsDisabled && !cm.IsPlayerReadyToEndTurn(Player)
               && rm.ActionQueueSet.IsEmpty && !rm.ActionExecutor.IsRunning;
    }

    Decision CombatDecision() => _build.Combat(State, Player);

    Decision? EventDecision(EventModel ev)
    {
        var d = _build.Event(State, Player, ev);
        if (d.Actions.Count == 0) return null;
        if (d.Actions.All(a => ((MegaCrit.Sts2.Core.Events.EventOption)a.Payload!).IsProceed))
        {
            // only "proceed" left: take it without asking
            RunManager.Instance.EventSynchronizer.ChooseLocalOption(d.Actions[0].Index);
            _roomDone = true;
            Settle();
            return null;
        }
        return d;
    }

    Decision RestDecision(RestSiteRoom rs) => _build.Rest(State, Player, rs.Options);

    Decision ShopDecision(MerchantRoom mr)
    {
        var inv = mr.GetLocalInventory();
        if (!_shopChecked)
        {
            _shopChecked = true;
            if (inv.CardRemovalEntry != null && !MegaCrit.Sts2.Core.Hooks.Hook.ShouldAllowMerchantCardRemoval(State, Player))
                inv.CardRemovalEntry.SetUsed();
        }
        return _build.Shop(State, Player, inv);
    }

    Decision? TreasureDecision(TreasureRoom tr)
    {
        if (!_treasureOpened)
        {
            _treasureOpened = true;
            _inFlight = OpenChest(tr);
            return null;
        }
        var relics = RunManager.Instance.TreasureRoomRelicSynchronizer.CurrentRelics;
        if (relics == null || relics.Count == 0)
        {
            _roomDone = true;
            return MapDecision();
        }
        return _build.Treasure(State, Player, relics);
    }

    async Task OpenChest(TreasureRoom tr)
    {
        await tr.DoNormalRewards();
    }

    Decision MapDecision() => _build.Map(State, Player);

    // ================================================================== execution

    void Execute(Decision d, int idx)
    {
        var a = d.Actions[idx];
        switch (a.Type)
        {
            case ActionType.PlayCard:
                if (!((CardModel)a.Payload!).TryManualPlay((Creature?)a.Target)) Trace?.Invoke("card play rejected");
                break;
            case ActionType.UsePotion:
                ((PotionModel)a.Payload!).EnqueueManualUse((Creature?)a.Target);
                if (!CombatManager.Instance.IsInProgress) RunManager.Instance.ActionExecutor.Unpause();
                break;
            case ActionType.DiscardPotion:
                RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new DiscardPotionGameAction(Player, (uint)a.Index, CombatManager.Instance.IsInProgress));
                if (!CombatManager.Instance.IsInProgress) RunManager.Instance.ActionExecutor.Unpause();
                break;
            case ActionType.EndTurn:
                RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new EndPlayerTurnAction(Player, Player.PlayerCombatState!.TurnNumber));
                break;
            case ActionType.ChooseOption when d.Type == DecisionType.Event:
                RunManager.Instance.EventSynchronizer.ChooseLocalOption(a.Index);
                break;
            case ActionType.ChooseOption when d.Type == DecisionType.Rest:
                _inFlight = RestChoose(a.Index);
                break;
            case ActionType.Proceed when d.Type is DecisionType.Rest or DecisionType.Shop:
                _roomDone = true;
                break;
            case ActionType.Buy:
            {
                var e = (MerchantEntry)a.Payload!;
                var mr = (MerchantRoom)State.CurrentRoom!;
                _inFlight = e.OnTryPurchaseWrapper(mr.GetLocalInventory());
                break;
            }
            case ActionType.RemoveCardService:
                _inFlight = BuyRemoval((MerchantCardRemovalEntry)a.Payload!);
                break;
            case ActionType.SelectRelic:
                RunManager.Instance.TreasureRoomRelicSynchronizer.PickRelicLocally(a.Index);
                _inFlight = AfterTreasure();
                break;
            case ActionType.Skip when d.Type == DecisionType.Treasure:
                RunManager.Instance.TreasureRoomRelicSynchronizer.SkipRelicLocally();
                _inFlight = AfterTreasure();
                break;
            case ActionType.ChooseMapNode:
                _inFlight = RunManager.Instance.EnterMapCoord(((MapPoint)a.Payload!).coord);
                break;
            case ActionType.Proceed when d.Type == DecisionType.Map:
                RunManager.Instance.ActChangeSynchronizer.SetLocalPlayerReady();
                RunManager.Instance.ActionExecutor.Unpause();
                break;
            default:
                Fail($"unhandled action {a.Type} in {d.Type}");
                break;
        }
        Settle();
    }

    async Task AfterCombat(CombatRoom cr)
    {
        if (Player.Creature.IsDead) return;
        if (cr.Encounter.ShouldGiveRewards && !_roomPrefinishedAtEntry)
            await cr.OfferRoomEndRewards();
        await RunManager.Instance.ProceedFromTerminalRewardsScreen();
    }

    async Task BuyRemoval(MerchantCardRemovalEntry e)
    {
        var mr = (MerchantRoom)State.CurrentRoom!;
        if (await e.OnTryPurchaseWrapper(mr.GetLocalInventory())) e.SetUsed();
    }

    async Task RestChoose(int index)
    {
        bool ok = await RunManager.Instance.RestSiteSynchronizer.ChooseLocalOption(index);
        if (ok) _roomDone = true;
    }

    async Task AfterTreasure()
    {
        await RunManager.Instance.ActionExecutor.FinishedExecutingActions();
        foreach (var t in _relicObtains.ToList()) await t;
        _relicObtains.Clear();
        if (State.CurrentRoom is TreasureRoom tr) await tr.DoExtraRewardsIfNeeded();
        _roomDone = true;
    }

    /// <summary>Replicates NTreasureRoomRelicCollection.OnRelicsAwarded (node-only in the game).</summary>
    void OnRelicsAwarded(List<MegaCrit.Sts2.Core.Entities.TreasureRelicPicking.RelicPickingResult> results)
    {
        foreach (var r in results)
        {
            if (r.type != MegaCrit.Sts2.Core.Entities.TreasureRelicPicking.RelicPickingResultType.Skipped && r.player != null)
                _relicObtains.Add(RelicCmd.Obtain(r.relic.ToMutable(), r.player));
        }
    }

    /// <summary>Crystal Sphere minigame: simple heuristic (big tool on the cell revealing most fog).</summary>
    async Task PlaySphere(MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame g)
    {
        var rng = new Random(State.Rng.Seed.GetHashCode() ^ State.TotalFloor);
        int w = g.cells.GetLength(0), hgt = g.cells.GetLength(1);
        int guard = 0;
        while (!g.IsFinished && guard++ < 200)
        {
            g.SetTool(MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame.CrystalSphereToolType.Big);
            MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereCell? best = null;
            int bestScore = -1;
            for (int x = 0; x < w; x++)
            for (int y = 0; y < hgt; y++)
            {
                var c = g.cells[x, y];
                if (!c.IsHidden) continue;
                int score = 0;
                for (int dx = -1; dx <= 1; dx++)
                for (int dy = -1; dy <= 1; dy++)
                {
                    int xx = x + dx, yy = y + dy;
                    if (xx >= 0 && yy >= 0 && xx < w && yy < hgt && g.cells[xx, yy].IsHidden) score++;
                }
                score = score * 4 + rng.Next(4);
                if (score > bestScore) { bestScore = score; best = c; }
            }
            if (best == null) break;
            await g.CellClicked(best);
        }
        if (!g.IsFinished) g.ForceMinigameEnd();
    }

    // ================================================================== hooks called from inside game logic

    /// <summary>Rewards screen (combat end, treasure, events): claim rewards one at a time, then proceed.</summary>
    async Task OfferRewards(RewardsSet set)
    {
        if (Coop.Active) { await OfferRewardsCoop(set); return; }
        set.ThrowInTestIfRewardsNotTaken = false;
        var sync = RunManager.Instance.RewardsSetSynchronizer;
        var tried = new HashSet<Reward>(ReferenceEqualityComparer.Instance);
        while (true)
        {
            if (Player.Creature.IsDead) return;
            var open = set.Rewards.Where(r => !r.SuccessfullySelected && !tried.Contains(r)
                                              && !(r is PotionReward && !Player.HasOpenPotionSlots)).ToList();
            bool potionWaiting = set.Rewards.Any(r => r is PotionReward && !r.SuccessfullySelected && !tried.Contains(r)) && !Player.HasOpenPotionSlots;
            if (open.Count == 0 && !potionWaiting) break;
            // gold is never harmful: claim it without asking (a dominated choice the policy only adds noise to)
            if (open.FirstOrDefault(DecisionBuilder.AutoClaim) is { } auto)
            {
                tried.Add(auto);
                await sync.SelectLocalReward(auto);
                continue;
            }
            var waiting = potionWaiting
                ? set.Rewards.OfType<PotionReward>().Where(r => !r.SuccessfullySelected && r.Potion != null)
                : null;
            bool allowProceed = !set.DisallowSkipping || open.All(r => tried.Contains(r)) || open.Count == 0;
            var d = _build.Rewards(State, Player, open, waiting, allowProceed);
            int k = Ask(d);
            var act = d.Actions[k];
            if (act.Type == ActionType.Proceed) break;
            if (act.Type == ActionType.DiscardPotion)
            {
                await MegaCrit.Sts2.Core.Commands.PotionCmd.Discard((PotionModel)act.Payload!);
                continue;
            }
            var reward = (Reward)act.Payload!;
            tried.Add(reward);
            await sync.SelectLocalReward(reward);
        }
        if (!sync.IsRewardsSetCompleted(set))
            sync.SkipLocalRewardsSet();
    }

    /// <summary>Card selection prompts (hand/deck/generated cards): sequential picks with an optional confirm.</summary>
    public Task<IEnumerable<CardModel>> GetSelectedCards(IEnumerable<CardModel> options, int minSelect, int maxSelect)
    {
        var opts = options.ToList();
        if (Coop.Active && opts.Count > 0 && opts[0].Owner is { } owner) Player = owner;
        var picked = new List<CardModel>();
        maxSelect = Math.Min(maxSelect, opts.Count);
        minSelect = Math.Min(minSelect, maxSelect);
        while (picked.Count < maxSelect)
        {
            var d = _build.CardSelect(State, Player, opts, picked, minSelect, maxSelect);
            int k = Ask(d);
            var act = d.Actions[k];
            if (act.Type == ActionType.Confirm) break;
            picked.Add((CardModel)act.Payload!);
        }
        return Task.FromResult<IEnumerable<CardModel>>(picked);
    }

    int ChooseBundle(Player p, IReadOnlyList<IReadOnlyList<CardModel>> bundles)
    {
        if (Coop.Active) Player = p;
        var d = _build.Bundle(State, Player, bundles);
        return d.Actions[Ask(d)].Index;
    }

    /// <summary>Card reward: pick one card, or an alternative (skip / reroll / ...).</summary>
    public CardRewardSelection GetSelectedCardReward(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives)
    {
        if (Coop.Active && options.Count > 0 && options[0].Card.Owner is { } owner) Player = owner;
        var d = _build.CardReward(State, Player, options.Select(o => o.Card).ToList(), alternatives);
        int k = Ask(d);
        var act = d.Actions[k];
        _build.NoteCardRewardChoice(act);
        return act.Type == ActionType.SelectCard
            ? new CardRewardSelection { card = (CardModel)act.Payload! }
            : new CardRewardSelection { alternative = act.Payload as CardRewardAlternative };
    }
}

using MegaCrit.Sts2.Core.Rooms;
using Sts2Ai.Shared.Encoding;
using System.IO.Hashing;
using System.Runtime.InteropServices;

namespace Sts2Sim.Engine;

/// <summary>
/// Training worker: reads commands from stdin (16-byte frames: int32 cmd, int32 a, int64 b),
/// writes observations to the shared-memory slot and 1 byte "R" to stdout per observation.
/// </summary>
public sealed class Worker
{
    const int CmdReset = 1, CmdStep = 2, CmdQuit = 3;
    readonly RunDriver _driver;
    readonly Vocab _vocab;
    readonly ShmWriter _shm;
    readonly Stream _in = Console.OpenStandardInput();
    readonly Stream _out = Console.OpenStandardOutput();
    readonly byte[] _frame = new byte[16];
    readonly int _ascension;
    readonly int[] _characters;
    Random _rng = new(0);
    readonly bool _uniformPractice;
    int _combatId, _hpAtCombatStart, _character;
    int[] _team = Array.Empty<int>();
    // ---- co-op episodes (--coop-prob): a share of fresh runs are 2..coopMaxPlayers player co-op games, every
    // seat played by the policy (one decision stream, shared team reward; HP header fields are team totals)
    readonly double _coopProb;
    readonly int _coopMaxPlayers;
    bool _wasInCombat;
    MegaCrit.Sts2.Core.Combat.CombatState? _observedCombatState;

    // ---- practice restarts (--restart-prob): an episode may start from a point recorded in one of this
    // worker's earlier runs by replaying that run's choices, checked against the stored encoded observation.
    // Bosses decide most lost runs, but each run reaches them only once,
    // so they get little training signal otherwise.
    // Points are kept per situation (act x {act start, monster, elite, boss fight}: Sit = act*4 + kind).
    // Which situation an episode practises is chosen by the trainer from its failure map (reset argument
    // bits 12+); without that (legacy) the worker picks act-3 start / act-3 boss / act-2 boss itself.
    // Within a situation, points whose outcome is most uncertain are replayed most (weight 0.1 + 4q(1-q),
    // q = smoothed loss rate): always-won and hopeless points carry little learning signal.
    //
    // Double-boss options (route 1, all off by default):
    //   --boss2-situation 1      the second final boss of a double-boss act (A10 act 3) gets its own situation
    //                            12 "a3_boss2" (13 situations) instead of sharing a3_boss with the first one;
    //   --boss2-hp-stratified 1  a3_boss2 points are drawn by entry-HP quarter: a non-empty quarter uniformly,
    //                            then a point within it as usual, so low-HP entries are practised too;
    //   --fight-outcome 1        final-boss points are scored by their fight, not the run: a boss-1 point
    //                            succeeds if the episode reaches boss 2 with HP >= 50% of max (or wins), a
    //                            boss-2 point if boss 2 is won;
    //   --practice-repeats k     a drawn point is replayed by this worker's next k-1 requests for its situation
    //                            and is not evicted before that, so its loss rate q is estimated from data.
    sealed class SavePoint(int[] team, ulong seed, int[] prefix, int ascension, ulong observationHash,
        int boss = 0, int entryHp = 0, int entryMaxHp = 1)
    {
        public readonly int[] Team = team;
        public readonly int Ascension = ascension;
        public readonly ulong Seed = seed;
        public readonly int[] Prefix = prefix;
        public readonly ulong ObservationHash = observationHash;
        /// <summary>1/2: first/second final boss of a double-boss act; 0: any other point.</summary>
        public readonly int Boss = boss;
        public readonly int EntryHp = entryHp, EntryMaxHp = Math.Max(1, entryMaxHp);
        public int Tries, Losses, Draws;
        /// <summary>Entry-HP quarter: [0, .25), [.25, .5), [.5, .75), [.75, 1].</summary>
        public int HpBucket => Math.Clamp(EntryHp * 4 / EntryMaxHp, 0, 3);
        public double Weight
        {
            get { double q = (Losses + 1.0) / (Tries + 2.0); return 0.1 + 4 * q * (1 - q); }
        }
    }
    SavePoint? _current;
    public const int NSit = 12, MaxSit = 13, SitBoss2 = 12, SitStart = 0, PoolCap = 16;
    /// <summary>Boss-1 points succeed (fight-level outcome) if boss 2 is entered with at least this share of max HP.</summary>
    public const double Boss2HpTarget = 0.5;
    // legacy (worker-chosen) practice: act-3 start, act-3 boss, act-2 boss (with a3_boss2: the act-3 boss share split)
    static readonly (int sit, double w)[] LegacySits = { (2 * 4 + SitStart, 0.35), (2 * 4 + 3, 0.45), (1 * 4 + 3, 0.2) };
    static readonly (int sit, double w)[] LegacySitsBoss2 =
        { (2 * 4 + SitStart, 0.35), (2 * 4 + 3, 0.25), (SitBoss2, 0.2), (1 * 4 + 3, 0.2) };
    readonly double _restartProb;
    readonly int _nSit;              // 12, or 13 with --boss2-situation
    readonly bool _boss2HpStratified, _fightOutcome, _practiceTrace;
    readonly int _practiceRepeats;
    readonly List<SavePoint>[] _pools = Enumerable.Range(0, MaxSit).Select(_ => new List<SavePoint>()).ToArray();
    readonly SavePoint?[] _repeating = new SavePoint?[MaxSit];  // point being replayed --practice-repeats times
    // second final boss of this episode (fight-level outcome): seen, HP / max HP / boss kills at its first decision
    bool _boss2Seen;
    int _boss2EntryHp, _boss2EntryMaxHp, _boss2EntryBossKills;
    readonly List<int> _choices = new();
    int[]? _replay;
    int _replayPos, _restartKind, _savedCombatId, _savedActs;
    int _prefixActions, _requestedRestart;
    bool _replayFailed, _firstLive;
    ulong _seed;
    int _runAscension;

    public Worker(PumpContext pump, Vocab vocab, string slot, int ascension, string characters, double restartProb = 0,
        double coopProb = 0, int coopMaxPlayers = 3, string practiceSampler = "uncertainty",
        bool boss2Situation = false, bool boss2HpStratified = false, bool fightOutcome = false, int practiceRepeats = 1,
        bool practiceTrace = false)
    {
        if (practiceSampler is not ("uncertainty" or "uniform"))
            throw new ArgumentException("unknown practice sampler", nameof(practiceSampler));
        if (boss2HpStratified && !boss2Situation)
            throw new ArgumentException("--boss2-hp-stratified needs --boss2-situation 1");
        if (practiceRepeats < 1) throw new ArgumentException("--practice-repeats must be >= 1");
        _uniformPractice = practiceSampler == "uniform";
        _restartProb = restartProb;
        _nSit = boss2Situation ? MaxSit : NSit;
        _boss2HpStratified = boss2HpStratified;
        _fightOutcome = fightOutcome;
        _practiceRepeats = practiceRepeats;
        _practiceTrace = practiceTrace;
        _coopProb = coopProb;
        _coopMaxPlayers = Math.Clamp(coopMaxPlayers, 2, 4);
        if (coopProb > 0) StateEncoder.EncodeAllies = true;
        _driver = new RunDriver(pump, vocab);
        _vocab = vocab;
        _shm = new ShmWriter(slot);
        _ascension = ascension;
        _characters = characters == "all"
            ? new[] { 0, 1, 2, 3, 4 }
            : characters.Split(',').Select(RunDriver.CharacterIndex).ToArray();
        _driver.Policy = OnDecision;
    }

    (int cmd, int a, long b) Read()
    {
        int got = 0;
        while (got < 16)
        {
            int n = _in.Read(_frame, got, 16 - got);
            if (n <= 0) Environment.Exit(0);
            got += n;
        }
        return (BitConverter.ToInt32(_frame, 0), BitConverter.ToInt32(_frame, 4), BitConverter.ToInt64(_frame, 8));
    }

    void Notify()
    {
        _out.WriteByte((byte)'R');
        _out.Flush();
    }

    (int hp, int maxHp) TeamHp()
    {
        bool coop = _driver.Players.Count > 1;
        var p = _driver.Player;
        return (coop ? _driver.Players.Sum(x => x.Creature.CurrentHp) : p.Creature.CurrentHp,
                coop ? _driver.Players.Sum(x => x.Creature.MaxHp) : p.Creature.MaxHp);
    }

    /// <summary>In the fight at the second boss node of a double-boss act (A10 act 3).</summary>
    bool InSecondBoss()
    {
        var st = _driver.State;
        var second = st.Map?.SecondBossMapPoint;
        return second != null && ReferenceEquals(st.CurrentMapPoint, second) && st.CurrentRoom is CombatRoom { RoomType: RoomType.Boss };
    }

    void FillHeader(bool done)
    {
        var h = _shm.Header;
        var st = _driver.State;
        var p = _driver.Player;
        bool inCombat = MegaCrit.Sts2.Core.Combat.CombatManager.Instance.IsInProgress;
        bool coop = _driver.Players.Count > 1;
        var (hp, maxHp) = TeamHp();
        // Consecutive combats can have no exposed non-combat decision between them
        // (notably the A10 final bosses). Track the actual combat-state identity,
        // rather than relying only on an observed false -> true flag transition.
        var combatState = inCombat ? MegaCrit.Sts2.Core.Combat.CombatManager.Instance.DebugOnlyGetState() : null;
        if (inCombat && (!_wasInCombat || !ReferenceEquals(combatState, _observedCombatState)))
        {
            _combatId++;
            _hpAtCombatStart = hp;
            if (!_boss2Seen && InSecondBoss())
            {
                _boss2Seen = true;
                (_boss2EntryHp, _boss2EntryMaxHp, _boss2EntryBossKills) = (hp, Math.Max(1, maxHp), _driver.BossKills);
            }
        }
        _observedCombatState = combatState;
        _wasInCombat = inCombat;
        h[ShmLayout.HDone] = done ? 1 : 0;
        h[ShmLayout.HWon] = _driver.Won ? 1 : 0;
        h[ShmLayout.HAct] = st.CurrentActIndex;
        h[ShmLayout.HActFloor] = st.ActFloor;
        h[ShmLayout.HTotalFloor] = st.TotalFloor;
        h[ShmLayout.HHp] = hp;
        h[ShmLayout.HMaxHp] = maxHp;
        h[ShmLayout.HCoop] = coop ? _driver.Players.Count : 0;
        h[ShmLayout.HGold] = p.Gold;
        h[ShmLayout.HInCombat] = inCombat ? 1 : 0;
        h[ShmLayout.HCombatId] = _combatId;
        h[ShmLayout.HDeckSize] = p.Deck.Cards.Count;
        h[ShmLayout.HError] = _driver.Error != null || _replayFailed ? 1 : 0;
        h[ShmLayout.HRestart] = _restartKind;
        h[ShmLayout.HPrefixActions] = _prefixActions;
        h[ShmLayout.HRequestedRestart] = _requestedRestart;
        int available = 0;
        for (int s = 0; s < _nSit; s++) if (_pools[s].Count > 0) available |= 1 << s;
        h[ShmLayout.HAvailableRestarts] = available;
        h[ShmLayout.HCharacter] = _character;
        h[ShmLayout.HAscension] = st.AscensionLevel;
        h[ShmLayout.HBossKills] = _driver.BossKills;
        h[ShmLayout.HEliteKills] = _driver.EliteKills;
        h[ShmLayout.HCombatHpStart] = _hpAtCombatStart;
        h[ShmLayout.HStepInRun] = _driver.Steps;
        h[ShmLayout.HCombatsWon] = _driver.CombatsWon;
        var room = st.CurrentRoom;
        h[ShmLayout.HRoomType] = (int)(room?.RoomType ?? MegaCrit.Sts2.Core.Rooms.RoomType.Unassigned);
        h[ShmLayout.HRoomModel] = room switch
        {
            MegaCrit.Sts2.Core.Rooms.CombatRoom cr => _vocab.Id(cr.Encounter),
            MegaCrit.Sts2.Core.Rooms.EventRoom er => _vocab.Id(er.CanonicalEvent),
            _ => 0,
        };
    }

    int OnDecision(Sts2Ai.Shared.Decisions.Decision d, StateEncoder enc)
    {
        if (_replay != null)
        {
            if (_replayFailed) return 0;
            if (_replayPos < _replay.Length)
            {
                int r = _replay[_replayPos++];
                if (r >= d.Actions.Count)
                {
                    // never seen so far: the replay left the recorded run; drop the episode (counted as an error)
                    _replayFailed = true;
                    _driver.AbortRequested = true;
                    return 0;
                }
                _choices.Add(r);
                _prefixActions++;
                return r;
            }
            if (_current != null && HashObservation(enc.B) != _current.ObservationHash)
            {
                _replayFailed = true;
                _driver.AbortRequested = true;
                Console.Error.WriteLine($"replay observation mismatch seed={_current.Seed} situation={_restartKind}");
                return 0;
            }
            _replay = null;
        }
        FillHeader(false);
        if (_restartProb > 0) RecordSavePoints();
        _shm.Write(enc.B);
        Notify();
        while (true)
        {
            var (cmd, a, _) = Read();
            if (cmd == CmdStep) { _choices.Add(a); return a; }
            if (cmd == CmdQuit) Environment.Exit(0);
            // a RESET in the middle of a run abandons it
            if (cmd == CmdReset) throw new OperationCanceledException("reset");
        }
    }

    void RecordSavePoints()
    {
        var st = _driver.State;
        bool inCombat = MegaCrit.Sts2.Core.Combat.CombatManager.Instance.IsInProgress;
        int act = Math.Clamp(st.CurrentActIndex, 0, 2);
        // fight kind 1..3 (monster, elite, boss); event fights are not practised
        int fight = inCombat && st.CurrentRoom is CombatRoom cr && cr.RoomType is RoomType.Monster or RoomType.Elite or RoomType.Boss
            ? (int)cr.RoomType : 0;
        if (_firstLive)
        {
            // first live decision of a restarted episode: its own start point is already in the pool
            _firstLive = false;
            _savedActs = (1 << (act + 1)) - 1;
            if (inCombat) _savedCombatId = _combatId;
            return;
        }
        if (!inCombat && (_savedActs & (1 << act)) == 0)
        {
            _savedActs |= 1 << act;
            Save(act * 4 + SitStart);
        }
        if (fight != 0 && _combatId != _savedCombatId)
        {
            _savedCombatId = _combatId;
            // final bosses of a double-boss act: boss 1 / boss 2 (boss 2 in its own situation with --boss2-situation)
            int boss = fight == 3 && act == 2 && st.Map?.SecondBossMapPoint != null ? (InSecondBoss() ? 2 : 1) : 0;
            Save(boss == 2 && _nSit > SitBoss2 ? SitBoss2 : act * 4 + fight, boss);
        }
    }

    void Save(int sit, int boss = 0)
    {
        var pool = _pools[sit];
        if (pool.Count >= PoolCap)
        {
            // FIFO, except that a point still being repeated (--practice-repeats) is kept until its repeats are done
            int i = _repeating[sit] != null && ReferenceEquals(pool[0], _repeating[sit]) ? 1 : 0;
            pool.RemoveAt(i);
        }
        var (hp, maxHp) = TeamHp();
        pool.Add(new SavePoint(_team, _seed, _choices.ToArray(), _runAscension, HashObservation(_driver.Enc.B), boss, hp, maxHp));
    }

    static ulong HashObservation(ObsBuilder b)
    {
        var h = new XxHash64();
        Span<int> sizes = stackalloc int[] { b.NumTokens, b.NumActions };
        h.Append(MemoryMarshal.AsBytes(sizes));
        h.Append(MemoryMarshal.AsBytes(b.TokCat.AsSpan(0, b.NumTokens * ObsSpec.NTokCat)));
        h.Append(MemoryMarshal.AsBytes(b.TokNum.AsSpan(0, b.NumTokens * ObsSpec.FTok)));
        h.Append(MemoryMarshal.AsBytes(b.ActCat.AsSpan(0, b.NumActions * ObsSpec.NActCat)));
        h.Append(MemoryMarshal.AsBytes(b.ActRef.AsSpan(0, b.NumActions * 2)));
        h.Append(MemoryMarshal.AsBytes(b.ActNum.AsSpan(0, b.NumActions * ObsSpec.FAct)));
        return h.GetCurrentHashAsUInt64();
    }

    /// <summary>The point a practice episode of this situation starts from: the point still being repeated
    /// (--practice-repeats), else a new draw.</summary>
    SavePoint? Draw(int sit)
    {
        var rep = _repeating[sit];
        if (rep != null && _pools[sit].Contains(rep))
        {
            if (++rep.Draws >= _practiceRepeats) _repeating[sit] = null;
            return rep;
        }
        _repeating[sit] = null;
        var sp = PickFrom(sit);
        if (sp != null)
        {
            sp.Draws = 1;  // draws in the current streak
            if (_practiceRepeats > 1) _repeating[sit] = sp;
        }
        return sp;
    }

    SavePoint? PickFrom(int sit)
    {
        var all = _pools[sit];
        if (all.Count == 0) return null;
        IReadOnlyList<SavePoint> pool = all;
        if (_boss2HpStratified && sit == SitBoss2)
        {
            // entry-HP strata: a non-empty HP quarter uniformly, then a point within it
            int mask = 0;
            foreach (var p in all) mask |= 1 << p.HpBucket;
            int k = _rng.Next(System.Numerics.BitOperations.PopCount((uint)mask)), bucket = 0;
            for (int b = 0; b < 4; b++)
                if ((mask >> b & 1) != 0 && k-- == 0) { bucket = b; break; }
            pool = all.Where(p => p.HpBucket == bucket).ToList();
        }
        if (_uniformPractice) return pool[_rng.Next(pool.Count)];
        double tw = 0;
        foreach (var p in pool) tw += p.Weight;
        double v = _rng.NextDouble() * tw;
        foreach (var p in pool) { v -= p.Weight; if (v <= 0) return p; }
        return pool[^1];
    }

    /// <summary>Practice outcome of the episode just finished: the run, or (--fight-outcome) the final-boss fight.</summary>
    bool PracticeSucceeded(SavePoint sp)
    {
        if (_driver.Won || !_fightOutcome || sp.Boss == 0) return _driver.Won;
        if (sp.Boss == 2) return _boss2Seen && _driver.BossKills > _boss2EntryBossKills;
        return _boss2Seen && _boss2EntryHp >= Boss2HpTarget * _boss2EntryMaxHp;
    }

    SavePoint? PickLegacy(out int sit)
    {
        sit = -1;
        var sits = _nSit > SitBoss2 ? LegacySitsBoss2 : LegacySits;
        double total = 0;
        foreach (var (s, w) in sits) if (_pools[s].Count > 0) total += w;
        if (total <= 0) return null;
        double u = _rng.NextDouble() * total;
        foreach (var (s, w) in sits)
        {
            if (_pools[s].Count == 0) continue;
            sit = s;
            u -= w;
            if (u <= 0) break;
        }
        return Draw(sit);
    }

    public void Run()
    {
        while (true)
        {
            var (cmd, a, b) = Read();
            if (cmd == CmdQuit) return;
            if (cmd != CmdReset) continue;
            // reset argument: a character (0..4, <0 random), or (ascension+1)*8 + character (7: random) to set the
            // episode's ascension (curriculum chosen by the trainer); bits 12+ (failure-map focus): 0 = the worker
            // decides about practice itself (legacy), 1 = a fresh run, s+2 = practise situation s
            // Reset command seed also controls character/restart sampling, independently of game RNG.
            _rng = new Random(unchecked((int)((ulong)b ^ ((ulong)b >> 32))));
            int sitField = a >= 0 ? a >> 12 : 0;
            _requestedRestart = sitField >= 2 ? Math.Min(sitField - 1, _nSit) : 0;
            if (a >= 0) a &= 0xFFF;
            int ascension = _ascension;
            if (a >= 8) { ascension = a / 8 - 1; a %= 8; if (a == 7) a = -1; }
            int character = a >= 0 && a < 5 ? a : _characters[_rng.Next(_characters.Length)];
            int[] team = new[] { character };
            if (_coopProb > 0 && _rng.NextDouble() < _coopProb)
            {
                team = new int[_rng.Next(2, _coopMaxPlayers + 1)];
                for (int i = 0; i < team.Length; i++) team[i] = _characters[_rng.Next(_characters.Length)];
            }
            ulong seed = (ulong)b;
            _combatId = 0;
            _wasInCombat = false;
            _observedCombatState = null;
            _choices.Clear();
            _replay = null;
            _replayPos = _restartKind = 0;
            _replayFailed = _firstLive = false;
            _savedActs = 0;
            _savedCombatId = -1;
            _current = null;
            _boss2Seen = false;
            _boss2EntryHp = _boss2EntryBossKills = 0;
            _boss2EntryMaxHp = 1;
            int sit = -1;
            SavePoint? sp = sitField >= 2 ? Draw(sit = Math.Min(sitField - 2, _nSit - 1))
                : sitField == 0 && _restartProb > 0 && _rng.NextDouble() < _restartProb ? PickLegacy(out sit)
                : null;
            if (sp != null)
            {
                _current = sp;
                team = sp.Team;
                seed = sp.Seed;
                ascension = sp.Ascension;
                _replay = sp.Prefix;
                _restartKind = sit + 1;  // header: situation + 1 (0 = fresh run)
                _firstLive = true;
            }
            _team = team;
            _character = team[0];
            _seed = seed;
            _runAscension = ascension;
            try
            {
                _driver.StartRun(team, ascension, seed);
                _driver.PlayToEnd();
            }
            catch (Exception e)
            {
                Console.Error.WriteLine("run error: " + e);
            }
            if (_current != null && !_replayFailed && _driver.Error == null)
            {
                bool success = PracticeSucceeded(_current);
                _current.Tries++;
                if (!success) _current.Losses++;
                if (_practiceTrace)
                    Console.Error.WriteLine($"practice sit={_restartKind - 1} boss={_current.Boss} entry_hp={_current.EntryHp}/{_current.EntryMaxHp} " +
                        $"draws={_current.Draws} tries={_current.Tries} losses={_current.Losses} success={(success ? 1 : 0)} won={(_driver.Won ? 1 : 0)} " +
                        $"boss2_seen={(_boss2Seen ? 1 : 0)} boss2_entry_hp={_boss2EntryHp}/{_boss2EntryMaxHp} boss_kills={_driver.BossKills} " +
                        $"prefix={_current.Prefix.Length} pool={_pools[_restartKind - 1].Count} " +
                        $"pool_quarters={string.Join(",", Enumerable.Range(0, 4).Select(q => _pools[_restartKind - 1].Count(x => x.HpBucket == q)))} " +
                        $"bucket={_current.HpBucket}");
            }
            else if (_current != null)
            {
                // a point whose replay failed is not repeated
                for (int s = 0; s < MaxSit; s++) if (ReferenceEquals(_repeating[s], _current)) _repeating[s] = null;
            }
            FillHeader(true);
            if (_driver.Error != null) Console.Error.WriteLine("run error: " + _driver.Error);
            _shm.WriteHeaderOnly();
            Notify();
            _driver.EndRun();
        }
    }
}

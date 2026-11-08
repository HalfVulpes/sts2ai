using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace Sts2Sim.Engine;

public sealed partial class RunDriver
{
    /// <summary>
    /// Start from a saved run state (the game's SerializableRun, e.g. the room-entry state the live game's
    /// CombatReplayWriter records): the run is restored with its RNG counters and re-enters the latest map point,
    /// exactly as the game's "continue" does. Single-player only.
    /// </summary>
    public void StartFromSave(SerializableRun save)
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
        State = RunState.FromSerializable(save);
        Players = State.Players.ToArray();
        Player = Players[0];
        Coop.Active = false;
        Coop.Players = Players;
        Coop.Pending.Clear();
        Await(RunManager.Instance.SetUpSavedSingleplayer(State, save));
        RunManager.Instance.TreasureRoomRelicSynchronizer.RelicsAwarded += OnRelicsAwarded;
        _sphere = null;
        RunManager.Instance.Launch();
        Await(RunManager.Instance.GenerateMap()); // restores the act's saved map (SavedMapsToLoad), as "continue" does
        var load = RunManager.Instance.LoadIntoLatestMapCoord(null);
        Settle();
        if (load.IsFaulted) throw new InvalidOperationException("loading the saved run failed: " + load.Exception?.GetBaseException());
        _inFlight = load;
    }
}

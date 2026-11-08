using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using Sts2Ai.Shared.Decisions;
using Sts2Ai.Shared.Encoding;

namespace Sts2Sim.Engine;

/// <summary>
/// Co-op runs (several players, all driven here; see <see cref="Coop"/>). Every decision belongs to one
/// player: <see cref="Player"/> is set to that player before the policy is asked.
///  * combat: players act one at a time; a player keeps acting until they end their turn;
///  * events / rest sites / shops: each player decides independently (shared events: votes);
///  * treasure: each player votes for a relic; map: each player votes for the next node; act change: each
///    player readies up. The game resolves the votes itself (host logic).
/// </summary>
public sealed partial class RunDriver
{
    public IReadOnlyList<Player> Players { get; private set; } = Array.Empty<Player>();

    bool[] _pDone = Array.Empty<bool>();       // this room: player finished (event / rest site / shop)
    bool[] _pShopChecked = Array.Empty<bool>();
    bool[] _actVoted = Array.Empty<bool>();
    int _actVotesAct = -1;
    bool _treasureVoted;

    int Slot(Player p)
    {
        for (int i = 0; i < Players.Count; i++) if (ReferenceEquals(Players[i], p)) return i;
        throw new InvalidOperationException("player not in run");
    }

    void ResetCoopRun()
    {
        _pDone = new bool[Players.Count];
        _pShopChecked = new bool[Players.Count];
        _actVoted = new bool[Players.Count];
        _actVotesAct = -1;
        _treasureVoted = false;
    }

    void CheckOverCoop()
    {
        if (CombatManager.Instance.IsInProgress && Players.Any(p => p.PlayerCombatState is { TurnNumber: > MaxCombatTurns }))
        {
            Stalemate = true;
            Over = true;
            Won = false;
            return;
        }
        if (Players.All(p => p.Creature.IsDead) || State.IsGameOver)
        {
            Over = true;
            Won = Won || RunVictory();
        }
    }

    /// <summary>A game task (room entry, act change, reward offer) is still running: wait for it.</summary>
    bool CoopBusy()
    {
        for (int i = Coop.Pending.Count - 1; i >= 0; i--)
        {
            var t = Coop.Pending[i];
            if (!t.IsCompleted) return true;
            Coop.Pending.RemoveAt(i);
            if (t.IsFaulted) Fail("task faulted: " + t.Exception);
        }
        var rm = RunManager.Instance;
        return !rm.ActionQueueSet.IsEmpty || rm.ActionExecutor.IsRunning;
    }

    Decision? DetectTopLevelCoop()
    {
        var room = State.CurrentRoom;
        if (!ReferenceEquals(room, _room))
        {
            _room = room;
            Trace?.Invoke($"ROOM act={State.CurrentActIndex + 1} floor={State.TotalFloor} {room?.RoomType} {(room as CombatRoom)?.Encounter.Id.Entry ?? (room as EventRoom)?.CanonicalEvent.Id.Entry}");
            _roomDone = false;
            _treasureOpened = false;
            _treasureVoted = false;
            Array.Clear(_pDone);
            Array.Clear(_pShopChecked);
            _roomPrefinishedAtEntry = room?.IsPreFinished == true && !CombatManager.Instance.IsInProgress && room is not CombatRoom { CombatState: not null } ? room.IsPreFinished : false;
        }
        if (_actVotesAct != State.CurrentActIndex)
        {
            _actVotesAct = State.CurrentActIndex;
            Array.Clear(_actVoted);
        }
        if (_sphere != null)
        {
            if (!_sphere.IsFinished) { _inFlight = PlaySphere(_sphere); _sphere = null; return null; }
            _sphere = null;
        }
        if (CombatManager.Instance.IsInProgress)
        {
            var actor = NextCombatActor();
            if (actor == null) return null;
            Player = actor;
            return CombatDecision();
        }
        if (_inFlight != null && !_inFlight.IsCompleted) return null;
        if (CoopBusy()) return null;
        var events = RunManager.Instance.EventSynchronizer;
        if (room is EventRoom { IsVictoryRoom: true })
        {
            // final dialogue: advance every player's copy until the game declares victory
            bool any = false;
            foreach (var p in Players)
            {
                var ev = events.GetEventForPlayer(p);
                if (ev.IsFinished || (events.IsShared && events.GetPlayerVote(p) != null)) continue;
                int idx = -1;
                for (int i = 0; i < ev.CurrentOptions.Count; i++) if (!ev.CurrentOptions[i].IsLocked) { idx = i; break; }
                if (idx < 0) continue;
                Coop.ChooseEvent(p, idx);
                any = true;
            }
            if (!any || State.IsGameOver || RunManager.Instance.WinTime > 0) { Won = true; Over = true; }
            Settle();
            _progressed = true;
            return null;
        }
        switch (room)
        {
            case EventRoom when !_roomDone:
            {
                bool waiting = false;
                for (int i = 0; i < Players.Count; i++)
                {
                    if (_pDone[i]) continue;
                    var p = Players[i];
                    var ev = events.GetEventForPlayer(p);
                    if (ev.IsFinished) { _pDone[i] = true; continue; }
                    if (events.IsShared && events.GetPlayerVote(p) != null) { waiting = true; continue; }
                    Player = p;
                    var d = _build.Event(State, p, ev);
                    if (d.Actions.Count == 0) { _pDone[i] = true; continue; }
                    if (d.Actions.All(a => ((MegaCrit.Sts2.Core.Events.EventOption)a.Payload!).IsProceed))
                    {
                        Coop.ChooseEvent(p, d.Actions[0].Index);
                        if (!events.IsShared) _pDone[i] = true;
                        Settle();
                        _progressed = true;
                        return null;
                    }
                    return d;
                }
                if (waiting) return null;
                _roomDone = true;
                break;
            }
            case RestSiteRoom when !_roomDone:
            {
                for (int i = 0; i < Players.Count; i++)
                {
                    if (_pDone[i]) continue;
                    var p = Players[i];
                    Player = p;
                    return _build.Rest(State, p, RunManager.Instance.RestSiteSynchronizer.GetOptionsForPlayer(p));
                }
                _roomDone = true;
                break;
            }
            case MerchantRoom mr when !_roomDone:
            {
                for (int i = 0; i < Players.Count; i++)
                {
                    if (_pDone[i]) continue;
                    var p = Players[i];
                    var inv = mr.Inventories[i];
                    if (!_pShopChecked[i])
                    {
                        _pShopChecked[i] = true;
                        if (inv.CardRemovalEntry != null && !MegaCrit.Sts2.Core.Hooks.Hook.ShouldAllowMerchantCardRemoval(State, p))
                            inv.CardRemovalEntry.SetUsed();
                    }
                    Player = p;
                    return _build.Shop(State, p, inv);
                }
                _roomDone = true;
                break;
            }
            case TreasureRoom tr when !_roomDone:
            {
                if (!_treasureOpened)
                {
                    _treasureOpened = true;
                    _inFlight = OpenChest(tr);
                    return null;
                }
                var trs = RunManager.Instance.TreasureRoomRelicSynchronizer;
                var relics = trs.CurrentRelics;
                if (relics == null || relics.Count == 0)
                {
                    // voting over (relics awarded) -> extra rewards; or no relics at all
                    _roomDone = true;
                    if (_treasureVoted) { _inFlight = AfterTreasure(); return null; }
                    break;
                }
                foreach (var p in Players)
                {
                    if (trs.GetPlayerVote(p).voteReceived) continue;
                    Player = p;
                    return _build.Treasure(State, p, relics);
                }
                return null;
            }
            case CombatRoom cr when !_roomDone:
                _roomDone = true;
                _inFlight = AfterCombatCoop(cr);
                return null;
        }
        return MapDecisionCoop();
    }

    /// <summary>The player who acts next in combat: the first one (in slot order) still in their play phase.</summary>
    Player? NextCombatActor()
    {
        var cm = CombatManager.Instance;
        var rm = RunManager.Instance;
        if (cm.IsOverOrEnding || !rm.ActionQueueSet.IsEmpty || rm.ActionExecutor.IsRunning) return null;
        // keep the current actor while they are still playing (a player plays out their turn)
        if (CanAct(Player)) return Player;
        foreach (var p in Players) if (CanAct(p)) return p;
        return null;

        bool CanAct(Player p) => p.Creature.IsAlive && p.PlayerCombatState is { Phase: PlayerTurnPhase.Play } && !cm.IsPlayerReadyToEndTurn(p);
    }

    Decision? MapDecisionCoop()
    {
        var mss = RunManager.Instance.MapSelectionSynchronizer;
        for (int i = 0; i < Players.Count; i++)
        {
            var p = Players[i];
            Player = p;
            var d = _build.Map(State, p);
            bool nextAct = d.Actions.Any(a => a.Type == ActionType.Proceed);
            if (nextAct ? _actVoted[i] : mss.GetVote(p) != null) continue;
            return d;
        }
        return null; // everyone voted: the game moves on
    }

    void ExecuteCoop(Decision d, int idx)
    {
        var a = d.Actions[idx];
        var p = Player;
        int slot = Slot(p);
        var rm = RunManager.Instance;
        switch (a.Type)
        {
            case ActionType.ChooseOption when d.Type == DecisionType.Event:
                Coop.ChooseEvent(p, a.Index);
                break;
            case ActionType.ChooseOption when d.Type == DecisionType.Rest:
                _inFlight = RestChooseCoop(p, slot, a.Index);
                break;
            case ActionType.Proceed when d.Type is DecisionType.Rest or DecisionType.Shop:
                _pDone[slot] = true;
                break;
            case ActionType.Buy:
                _inFlight = ((MerchantEntry)a.Payload!).OnTryPurchaseWrapper(((MerchantRoom)State.CurrentRoom!).Inventories[slot]);
                break;
            case ActionType.RemoveCardService:
                _inFlight = BuyRemovalCoop((MerchantCardRemovalEntry)a.Payload!, slot);
                break;
            case ActionType.SelectRelic:
                _treasureVoted = true;
                rm.ActionQueueSynchronizer.RequestEnqueue(new PickRelicAction(p, a.Index));
                break;
            case ActionType.Skip when d.Type == DecisionType.Treasure:
                _treasureVoted = true;
                rm.ActionQueueSynchronizer.RequestEnqueue(new PickRelicAction(p, null));
                break;
            case ActionType.ChooseMapNode:
            {
                var vote = new MapVote { coord = ((MapPoint)a.Payload!).coord, mapGenerationCount = rm.MapSelectionSynchronizer.MapGenerationCount };
                rm.ActionQueueSynchronizer.RequestEnqueue(new VoteForMapCoordAction(p, State.MapLocation, vote));
                break;
            }
            case ActionType.Proceed when d.Type == DecisionType.Map:
                _actVoted[slot] = true;
                rm.ActionQueueSynchronizer.RequestEnqueue(new VoteToMoveToNextActAction(p));
                rm.ActionExecutor.Unpause();
                break;
            default:
                // combat inputs, potions: submitted as actions owned by the player, same as single-player
                Execute(d, idx);
                return;
        }
        Settle();
    }

    async Task RestChooseCoop(Player p, int slot, int index)
    {
        if (await Coop.ChooseRest(p, index)) _pDone[slot] = true;
    }

    async Task BuyRemovalCoop(MerchantCardRemovalEntry e, int slot)
    {
        if (await e.OnTryPurchaseWrapper(((MerchantRoom)State.CurrentRoom!).Inventories[slot])) e.SetUsed();
    }

    async Task AfterCombatCoop(CombatRoom cr)
    {
        if (cr.Encounter.ShouldGiveRewards && !_roomPrefinishedAtEntry)
            await cr.OfferRoomEndRewards();
        // every player's rewards screen must be finished before leaving (leaving skips what is left)
        while (Coop.Pending.Any(t => !t.IsCompleted))
            await Task.WhenAll(Coop.Pending.ToList());
        await RunManager.Instance.ProceedFromTerminalRewardsScreen();
    }

    /// <summary>One player's rewards screen (the game offers every player their own set).</summary>
    async Task OfferRewardsCoop(RewardsSet set)
    {
        set.ThrowInTestIfRewardsNotTaken = false;
        var p = set.Player;
        var sync = RunManager.Instance.RewardsSetSynchronizer;
        var tried = new HashSet<Reward>(ReferenceEqualityComparer.Instance);
        while (true)
        {
            if (p.Creature.IsDead) return;
            var open = set.Rewards.Where(r => !r.SuccessfullySelected && !tried.Contains(r)
                                              && !(r is PotionReward && !p.HasOpenPotionSlots)).ToList();
            bool potionWaiting = set.Rewards.Any(r => r is PotionReward && !r.SuccessfullySelected && !tried.Contains(r)) && !p.HasOpenPotionSlots;
            if (open.Count == 0 && !potionWaiting) break;
            if (open.FirstOrDefault(DecisionBuilder.AutoClaim) is { } auto)
            {
                tried.Add(auto);
                await Coop.SelectReward(p, set, auto);
                continue;
            }
            var waiting = potionWaiting
                ? set.Rewards.OfType<PotionReward>().Where(r => !r.SuccessfullySelected && r.Potion != null)
                : null;
            bool allowProceed = !set.DisallowSkipping || open.All(r => tried.Contains(r)) || open.Count == 0;
            Player = p;
            var d = _build.Rewards(State, p, open, waiting, allowProceed);
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
            await Coop.SelectReward(p, set, reward);
        }
        if (!sync.IsRewardsSetCompleted(set))
            Coop.SkipRewards(p);
    }
}

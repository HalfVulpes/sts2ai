using System.Reflection;
using System.Runtime.CompilerServices;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Models.Relics;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Multiplayer.Quality;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.Platform;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2Sim.Engine;

/// <summary>
/// Co-op (2-4 players) in one process: our process is the multiplayer host and every player's
/// input is submitted the way the host would receive it from that player's client.
///  * <see cref="LoopbackHostNetService"/>: a host whose messages go nowhere (all players are here);
///  * per-player choices go through the synchronizers' own per-player paths (events, rest site,
///    rewards) or as actions owned by that player (combat, map/act votes, relic votes);
///  * a few game methods decide "is this the local player's job?" with LocalContext.IsMe; inside those
///    (and only those) every player counts as local, because every player's client lives here:
///    reward offers, card-reward choice, relic/card selection prompts, merchant discount;
///  * each client normally reports "ready for the enemy turn" itself: the host's report is copied
///    for the other players;
///  * the end-turn button lock (a UI guard for the one local player) is skipped.
/// Everything is inactive unless <see cref="Active"/> (set by RunDriver for runs with > 1 player).
/// </summary>
public static class Coop
{
    public static bool Active;

    /// <summary>Game tasks started by the game itself that the driver must wait for (room entry, reward offers).</summary>
    public static readonly List<Task> Pending = new();

    public static void Apply(Harmony h)
    {
        var tr = new HarmonyMethod(typeof(Coop), nameof(IsMeTranspiler));
        foreach (var m in new MethodBase[]
                 {
                     AccessTools.AsyncMoveNext(AccessTools.Method(typeof(RewardsSet), nameof(RewardsSet.Offer))),
                     AccessTools.AsyncMoveNext(AccessTools.Method(typeof(CardReward), "OnSelect")),
                     AccessTools.Method(typeof(RelicSelectCmd), "ShouldSelectLocalRelic"),
                     AccessTools.Method(typeof(CardSelectCmd), "ShouldSelectLocalCard"),
                     AccessTools.Method(typeof(MembershipCard), nameof(MembershipCard.ModifyMerchantPrice)),
                 })
        {
            int before = _replaced;
            h.Patch(m, transpiler: tr);
            if (_replaced == before) throw new InvalidOperationException("coop: no LocalContext.IsMe call found in " + m.DeclaringType?.Name + "." + m.Name);
        }
        h.Patch(AccessTools.Method(typeof(CombatManager), nameof(CombatManager.OnEndedTurnLocally)),
            prefix: new HarmonyMethod(typeof(Coop), nameof(SkipWhenActive)));
        h.Patch(AccessTools.Method(typeof(ActionQueueSynchronizer), nameof(ActionQueueSynchronizer.RequestEnqueue)),
            postfix: new HarmonyMethod(typeof(Coop), nameof(RequestEnqueuePostfix)));
        // tasks the game starts with RunSafely (map move, act change, reward offers): the driver waits on them
        h.Patch(AccessTools.Method(typeof(RunManager), nameof(RunManager.EnterMapCoord)),
            postfix: new HarmonyMethod(typeof(Coop), nameof(TrackTask)));
        h.Patch(AccessTools.Method(typeof(RunManager), nameof(RunManager.EnterNextAct)),
            postfix: new HarmonyMethod(typeof(Coop), nameof(TrackTask)));
        h.Patch(AccessTools.Method(typeof(RewardsSet), nameof(RewardsSet.Offer)),
            postfix: new HarmonyMethod(typeof(Coop), nameof(TrackTask)));
        // Mend (heal an ally at a rest site) is chosen through UI targeting only: not offered yet
        h.Patch(AccessTools.PropertyGetter(typeof(RestSiteOption), nameof(RestSiteOption.IsEnabled)),
            prefix: new HarmonyMethod(typeof(Coop), nameof(MendDisabledWhenActive)));
    }

    // ------------------------------------------------------------------ patches

    static int _replaced;

    static IEnumerable<HarmonyLib.CodeInstruction> IsMeTranspiler(IEnumerable<HarmonyLib.CodeInstruction> code)
    {
        var isMeP = AccessTools.Method(typeof(LocalContext), nameof(LocalContext.IsMe), new[] { typeof(Player) });
        var isMeC = AccessTools.Method(typeof(LocalContext), nameof(LocalContext.IsMe), new[] { typeof(Creature) });
        foreach (var ins in code)
        {
            if (ins.Calls(isMeP)) { ins.operand = AccessTools.Method(typeof(Coop), nameof(IsMeOrCoop)); _replaced++; }
            else if (ins.Calls(isMeC)) { ins.operand = AccessTools.Method(typeof(Coop), nameof(IsMeOrCoopCreature)); _replaced++; }
            yield return ins;
        }
    }

    public static bool IsMeOrCoop(Player? p) => Active ? p != null : LocalContext.IsMe(p);
    public static bool IsMeOrCoopCreature(Creature? c) => Active ? c?.Player != null : LocalContext.IsMe(c);

    static bool SkipWhenActive() => !Active;

    static bool MendDisabledWhenActive(RestSiteOption __instance, ref bool __result)
    {
        if (!Active || __instance is not MendRestSiteOption) return true;
        __result = false;
        return false;
    }

    [ThreadStatic] static bool _propagating;

    static void RequestEnqueuePostfix(ActionQueueSynchronizer __instance, GameAction action)
    {
        if (!Active || _propagating || action is not ReadyToBeginEnemyTurnAction || action.OwnerId != LocalContext.NetId) return;
        _propagating = true;
        try
        {
            foreach (var p in Players)
                if (p.NetId != action.OwnerId)
                    __instance.RequestEnqueue(new ReadyToBeginEnemyTurnAction(p));
        }
        finally { _propagating = false; }
    }

    static void TrackTask(Task __result)
    {
        if (Active) Pending.Add(__result);
    }

    /// <summary>Players of the current co-op run (set by the driver).</summary>
    public static IReadOnlyList<Player> Players = Array.Empty<Player>();

    // ------------------------------------------------------------------ per-player inputs (host-side paths)

    [UnsafeAccessor(UnsafeAccessorKind.Method, Name = "ChooseOptionForEvent")]
    static extern void ChooseOptionForEvent(EventSynchronizer s, Player player, int optionIndex);
    [UnsafeAccessor(UnsafeAccessorKind.Method, Name = "PlayerVotedForSharedOptionIndex")]
    static extern void PlayerVotedForSharedOptionIndex(EventSynchronizer s, Player player, uint optionIndex, uint pageIndex);
    [UnsafeAccessor(UnsafeAccessorKind.Field, Name = "_pageIndex")]
    static extern ref uint EventPageIndex(EventSynchronizer s);
    [UnsafeAccessor(UnsafeAccessorKind.Method, Name = "ChooseOption")]
    static extern Task<bool> RestChooseOption(RestSiteSynchronizer s, Player player, int optionIndex);
    [UnsafeAccessor(UnsafeAccessorKind.Method, Name = "SelectRewardForPlayer")]
    static extern Task SelectRewardForPlayer(RewardsSetSynchronizer s, Player player, int rewardIndex);

    static readonly MethodInfo SkipRewardsTop = AccessTools.Method(typeof(RewardsSetSynchronizer), "SkipRewardsSetOnStackTopForPlayer");

    /// <summary>Event option for one player: a vote during shared events, the option itself otherwise.</summary>
    public static void ChooseEvent(Player p, int index)
    {
        var sync = RunManager.Instance.EventSynchronizer;
        if (sync.IsShared) PlayerVotedForSharedOptionIndex(sync, p, (uint)index, EventPageIndex(sync));
        else ChooseOptionForEvent(sync, p, index);
    }

    public static Task<bool> ChooseRest(Player p, int index) => RestChooseOption(RunManager.Instance.RestSiteSynchronizer, p, index);

    public static Task SelectReward(Player p, RewardsSet set, Reward r) =>
        SelectRewardForPlayer(RunManager.Instance.RewardsSetSynchronizer, p, set.Rewards.IndexOf(r));

    public static void SkipRewards(Player p) => SkipRewardsTop.Invoke(RunManager.Instance.RewardsSetSynchronizer, new object[] { p });

    // ------------------------------------------------------------------ net service

    /// <summary>A multiplayer host with no remote peers: every player is driven in this process.</summary>
    public sealed class LoopbackHostNetService(ulong netId) : INetGameService
    {
        bool _loading;
        public ulong NetId => netId;
        public bool IsConnected => true;
        public bool IsGameLoading => _loading;
        public NetGameType Type => NetGameType.Host;
        public PlatformType Platform => PlatformType.None;
        public event Action<NetErrorInfo>? Disconnected { add { } remove { } }
        public void SendMessage<T>(T message, ulong playerId) where T : INetMessage { }
        public void SendMessage<T>(T message) where T : INetMessage { }
        public void RegisterMessageHandler<T>(MessageHandlerDelegate<T> handler) where T : INetMessage { }
        public void UnregisterMessageHandler<T>(MessageHandlerDelegate<T> handler) where T : INetMessage { }
        public void Update() { }
        public void Disconnect(NetError reason, bool now = false) { }
        public ConnectionStats? GetStatsForPeer(ulong peerId) => new ConnectionStats(peerId);
        public void SetGameLoading(bool isLoading) => _loading = isLoading;
        public void SetBufferMessages(bool bufferMessages) { }
        public string? GetRawLobbyIdentifier() => null;
    }
}

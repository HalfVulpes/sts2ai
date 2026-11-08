using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using Sts2Ai.Shared.Encoding;

namespace Sts2Ai.Shared.Decisions;

/// <summary>
/// Builds every kind of decision (legal actions + encoded observation) from live game objects.
/// Shared by the training simulator (sim/Engine/RunDriver.cs) and the in-game mod, so a decision is
/// encoded identically in both. Each method encodes the full state first, then the offers / options.
/// </summary>
public sealed class DecisionBuilder(StateEncoder enc, Vocab vocab, RunHistorySummary history)
{
    public StateEncoder Enc => enc;

    void Begin(IRunState s, Player p, DecisionType t) => enc.EncodeState(s, p, t, history);

    static Dictionary<object, int> Refs() => new(ReferenceEqualityComparer.Instance);

    public Decision Combat(IRunState s, Player p)
    {
        var d = new Decision();
        CombatActions.Enumerate(p, vocab, d);
        Begin(s, p, DecisionType.Combat);
        ActionEncoder.Encode(enc, d);
        return d;
    }

    /// <summary>Event options that can be chosen (locked / already chosen ones are encoded but not offered).</summary>
    public Decision Event(IRunState s, Player p, EventModel ev)
    {
        var opts = ev.CurrentOptions;
        var d = new Decision { Type = DecisionType.Event };
        Begin(s, p, DecisionType.Event);
        var refs = Refs();
        for (int i = 0; i < opts.Count; i++)
        {
            var o = opts[i];
            refs[o] = OfferEncoding.EventOption(enc, ev, o, i);
            if (o.IsLocked || o.WasChosen) continue;
            d.Add(ActionType.ChooseOption, vocab.Id(ev), o.IsProceed ? 1 : 0, o, index: i, label: $"event {ev.Id.Entry}: {o.TextKey}");
        }
        ActionEncoder.Encode(enc, d, refs);
        return d;
    }

    /// <param name="usable">extra availability check (the mod: options that need UI targeting are unavailable, as in
    /// training where co-op Mend is disabled)</param>
    public Decision Rest(IRunState s, Player p, IReadOnlyList<RestSiteOption> opts, Func<RestSiteOption, bool>? usable = null)
    {
        var d = new Decision { Type = DecisionType.Rest };
        Begin(s, p, DecisionType.Rest);
        var refs = Refs();
        for (int i = 0; i < opts.Count; i++)
        {
            var o = opts[i];
            bool enabled = o.IsEnabled && (usable?.Invoke(o) ?? true);
            refs[o] = OfferEncoding.Option(enc, 0, "REST/" + o.OptionId, enabled ? 0 : 1, i);
            if (!enabled) continue;
            d.Add(ActionType.ChooseOption, vocab.Hash("REST/" + o.OptionId), 0, o, index: i, label: "rest " + o.OptionId);
        }
        d.Add(ActionType.Proceed, label: "leave rest site");
        ActionEncoder.Encode(enc, d, refs);
        return d;
    }

    public Decision Shop(IRunState s, Player p, MerchantInventory inv)
    {
        var d = new Decision { Type = DecisionType.Shop };
        Begin(s, p, DecisionType.Shop);
        var refs = Refs();
        int pos = 0;
        foreach (var e in inv.AllEntries)
        {
            if (!e.IsStocked) { pos++; continue; }
            int t, id;
            string label;
            switch (e)
            {
                case MerchantCardEntry ce when ce.CreationResult?.Card is { } card:
                    t = OfferEncoding.OfferCard(enc, card, pos, e.Cost, e.EnoughGold); id = vocab.Id(card); label = "buy card " + card.Id.Entry;
                    break;
                case MerchantRelicEntry re when re.Model is { } relic:
                    t = OfferEncoding.OfferRelic(enc, relic, pos, e.Cost, e.EnoughGold); id = vocab.Id(relic); label = "buy relic " + relic.Id.Entry;
                    break;
                case MerchantPotionEntry pe when pe.Model is { } potion:
                    t = OfferEncoding.OfferPotion(enc, potion, pos, e.Cost, e.EnoughGold); id = vocab.Id(potion); label = "buy potion " + potion.Id.Entry;
                    break;
                case MerchantCardRemovalEntry:
                    t = OfferEncoding.Option(enc, 0, "SHOP/REMOVE", 0, pos, e.Cost / 150f, e.EnoughGold ? 1 : 0); id = vocab.Hash("SHOP/REMOVE"); label = "card removal";
                    break;
                default:
                    pos++;
                    continue;
            }
            refs[e] = t;
            // a potion bought with a full belt, or when a relic forbids potions, fails without spending gold
            // and the item stays: a no-op the real UI also refuses; offering it would let a greedy policy loop
            bool usable = e is not MerchantPotionEntry pot
                          || (p.HasOpenPotionSlots && pot.Model != null
                              && MegaCrit.Sts2.Core.Hooks.Hook.ShouldProcurePotion(s, p.Creature.CombatState, pot.Model, p));
            if (e.EnoughGold && usable)
            {
                var a = d.Add(e is MerchantCardRemovalEntry ? ActionType.RemoveCardService : ActionType.Buy, id, 0, e, label: $"{label} ({e.Cost}g)");
                a.Num[0] = e.Cost / 150f;
            }
            pos++;
        }
        d.Add(ActionType.Proceed, label: "leave shop");
        ActionEncoder.Encode(enc, d, refs);
        return d;
    }

    public Decision Treasure(IRunState s, Player p, IReadOnlyList<RelicModel> relics)
    {
        var d = new Decision { Type = DecisionType.Treasure };
        Begin(s, p, DecisionType.Treasure);
        var refs = Refs();
        for (int i = 0; i < relics.Count; i++)
        {
            refs[relics[i]] = OfferEncoding.OfferRelic(enc, relics[i], i);
            d.Add(ActionType.SelectRelic, vocab.Id(relics[i]), 0, relics[i], index: i, label: "take relic " + relics[i].Id.Entry);
        }
        d.Add(ActionType.Skip, label: "skip relic");
        ActionEncoder.Encode(enc, d, refs);
        return d;
    }

    /// <summary>Next map node (or "next act" after the final boss of an act), plus usable potions.</summary>
    public Decision Map(RunState s, Player p)
    {
        var d = new Decision { Type = DecisionType.Map };
        Begin(s, p, DecisionType.Map);
        var refs = Refs();
        var map = s.Map;
        var cur = s.CurrentMapPoint;
        var current = s.CurrentRoom;
        bool bossDone = cur != null && (cur == map.BossMapPoint || cur == map.SecondBossMapPoint) && current is CombatRoom { RoomType: RoomType.Boss };
        bool needSecondBoss = bossDone && cur == map.BossMapPoint && map.SecondBossMapPoint != null;
        if (bossDone && !needSecondBoss)
        {
            d.Add(ActionType.Proceed, label: "next act");
        }
        else
        {
            IEnumerable<MapPoint> next = cur == null
                ? new[] { map.StartingMapPoint }
                : needSecondBoss ? new[] { map.SecondBossMapPoint! } : cur.Children;
            var stats = new OfferEncoding.PathStats();
            int i = 0;
            foreach (var mp in next.OrderBy(mp => mp.coord.col))
            {
                refs[mp] = OfferEncoding.MapNode(enc, mp, i++, stats);
                d.Add(ActionType.ChooseMapNode, vocab.Hash("MAP/" + mp.PointType), (int)mp.PointType, mp, label: $"map {mp.PointType} ({mp.coord.col},{mp.coord.row})");
            }
        }
        AddOutOfCombatPotions(p, d);
        ActionEncoder.Encode(enc, d, refs);
        return d;
    }

    void AddOutOfCombatPotions(Player p, Decision d)
    {
        int slot = 0;
        foreach (var pot in p.PotionSlots)
        {
            if (pot != null && CombatActions.CanUsePotionNow(pot, inCombat: false))
            {
                foreach (var t in CombatActions.CandidateTargets(pot.TargetType, p, null))
                    if (SafeValid(pot, t)) d.Add(ActionType.UsePotion, vocab.Id(pot), slot, pot, t, label: "potion " + pot.Id.Entry);
            }
            slot++;
        }
    }

    static bool SafeValid(PotionModel p, Creature? t) { try { return p.IsValidTarget(t); } catch { return false; } }

    /// <summary>Rewards that are always taken without asking the policy (both in training and in the game).</summary>
    public static bool AutoClaim(Reward r) => r is GoldReward;

    /// <summary>
    /// Rewards screen: claim one of the open rewards, drop a potion to make room for a waiting one
    /// (belt full), or proceed (when skipping is allowed).
    /// </summary>
    public Decision Rewards(IRunState s, Player p, IReadOnlyList<Reward> open, IEnumerable<PotionReward>? waitingPotions, bool allowProceed)
    {
        var d = new Decision { Type = DecisionType.Rewards };
        Begin(s, p, DecisionType.Rewards);
        var refs = Refs();
        for (int i = 0; i < open.Count; i++)
        {
            var r = open[i];
            refs[r] = OfferEncoding.Reward(enc, r, i);
            if (r is CardReward cr)
                foreach (var c in cr.Cards) OfferEncoding.OfferCard(enc, c, i);
            var a = d.Add(ActionType.ClaimReward, vocab.Hash("REWARD/" + r.GetType().Name), 0, r, label: "claim " + r.GetType().Name);
            if (r is GoldReward g) a.Num[0] = g.Amount / 100f;
        }
        if (waitingPotions != null)
        {
            foreach (var pr in waitingPotions)
                refs[pr] = OfferEncoding.OfferPotion(enc, pr.Potion!, 0);
            int slot = 0;
            foreach (var pot in p.PotionSlots)
            {
                if (pot != null && p.CanRemovePotions)
                    d.Add(ActionType.DiscardPotion, vocab.Id(pot), slot, pot, index: slot, label: "discard " + pot.Id.Entry + " for new potion");
                slot++;
            }
        }
        if (allowProceed)
            d.Add(ActionType.Proceed, label: "proceed (skip rest)");
        ActionEncoder.Encode(enc, d, refs);
        return d;
    }

    /// <summary>One pick of a (possibly multi-) card selection prompt; Confirm once enough are picked.</summary>
    public Decision CardSelect(IRunState s, Player p, IReadOnlyList<CardModel> opts, ICollection<CardModel> picked, int minSelect, int maxSelect)
    {
        var d = new Decision { Type = DecisionType.CardSelect, MinSelect = minSelect, MaxSelect = maxSelect };
        Begin(s, p, DecisionType.CardSelect);
        var refs = Refs();
        for (int i = 0; i < opts.Count; i++)
        {
            var c = opts[i];
            // cards already tokenized (hand cards) are referenced directly; others get offer tokens
            if (!enc.CardTok.ContainsKey(c)) OfferEncoding.OfferCard(enc, c, i);
            if (picked.Contains(c)) continue;
            var a = d.Add(ActionType.SelectCard, vocab.Id(c), Math.Min(picked.Count, 127), c, label: "select " + c.Id.Entry);
            a.Num[0] = minSelect / 5f;
            a.Num[1] = maxSelect / 5f;
            a.Num[2] = picked.Count / 5f;
        }
        if (picked.Count >= minSelect)
            d.Add(ActionType.Confirm, 0, Math.Min(picked.Count, 127), label: $"confirm {picked.Count} card(s)");
        ActionEncoder.Encode(enc, d, refs);
        return d;
    }

    public Decision Bundle(IRunState s, Player p, IReadOnlyList<IReadOnlyList<CardModel>> bundles)
    {
        var d = new Decision { Type = DecisionType.BundleSelect };
        Begin(s, p, DecisionType.BundleSelect);
        var refs = Refs();
        for (int i = 0; i < bundles.Count; i++)
        {
            refs[bundles[i]] = OfferEncoding.Option(enc, 0, "BUNDLE", 0, i, bundles[i].Count / 5f);
            foreach (var c in bundles[i]) OfferEncoding.OfferCard(enc, c, i);
            d.Add(ActionType.ChooseOption, vocab.Hash("BUNDLE"), i, bundles[i], index: i, label: "bundle " + string.Join(",", bundles[i].Select(c => c.Id.Entry)));
        }
        ActionEncoder.Encode(enc, d, refs);
        return d;
    }

    readonly System.Runtime.CompilerServices.ConditionalWeakTable<CardRewardAlternative, object> _usedAlternatives = new();

    /// <summary>Call with the action taken on a card reward: a reroll alternative is used up.</summary>
    public void NoteCardRewardChoice(LegalAction a)
    {
        if (a.Payload is CardRewardAlternative alt && alt.OptionId == "REROLL") _usedAlternatives.AddOrUpdate(alt, alt);
    }

    /// <summary>Card reward: take one card, or an alternative (skip / reroll / ...).</summary>
    public Decision CardReward(IRunState s, Player p, IReadOnlyList<CardModel> cards, IReadOnlyList<CardRewardAlternative> alternatives)
    {
        var d = new Decision { Type = DecisionType.CardReward };
        Begin(s, p, DecisionType.CardReward);
        for (int i = 0; i < cards.Count; i++)
        {
            var c = cards[i];
            OfferEncoding.OfferCard(enc, c, i);
            d.Add(ActionType.SelectCard, vocab.Id(c), 0, c, label: "take " + c.Id.Entry + (c.IsUpgraded ? "+" : ""));
        }
        for (int i = 0; i < alternatives.Count; i++)
        {
            // the game builds the alternatives once per reward and never re-checks CardReward.CanReroll,
            // so "REROLL" would stay on offer forever; the intended rule is one reroll per reward
            if (_usedAlternatives.TryGetValue(alternatives[i], out _)) continue;
            d.Add(ActionType.Skip, vocab.Hash("CARDALT/" + alternatives[i].OptionId), i, alternatives[i], label: "alt " + alternatives[i].OptionId);
        }
        if (alternatives.Count == 0) d.Add(ActionType.Skip, vocab.Hash("CARDALT/skip"), 0, label: "skip");
        ActionEncoder.Encode(enc, d);
        return d;
    }
}

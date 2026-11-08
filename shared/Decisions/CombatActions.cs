using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Models;
using Sts2Ai.Shared.Encoding;

namespace Sts2Ai.Shared.Decisions;

/// <summary>Enumerates the legal player inputs during the combat play phase.</summary>
public static class CombatActions
{
    public static void Enumerate(Player player, Vocab vocab, Decision d)
    {
        d.Type = DecisionType.Combat;
        var pcs = player.PlayerCombatState!;
        var combat = player.Creature.CombatState!;
        foreach (var card in pcs.Hand.Cards)
        {
            bool playable;
            try { playable = card.CanPlay(); } catch { playable = false; }
            if (!playable) continue;
            foreach (var target in CandidateTargets(card.TargetType, player, combat))
            {
                bool ok;
                try { ok = card.CanPlayTargeting(target); } catch { ok = false; }
                if (!ok) continue;
                var a = d.Add(ActionType.PlayCard, vocab.Id(card), (int)card.TargetType, card, target,
                    label: $"play {card.Id.Entry}{(card.IsUpgraded ? "+" : "")}{(target != null ? " -> " + TargetLabel(target, player) : "")}");
                try { a.Num[0] = card.EnergyCost.CostsX ? pcs.Energy / 3f : card.EnergyCost.GetResolved() / 3f; } catch { }
            }
        }
        int slot = 0;
        foreach (var potion in player.PotionSlots)
        {
            if (potion != null)
            {
                if (CanUsePotionNow(potion, inCombat: true))
                {
                    foreach (var target in PotionTargets(potion.TargetType, player, combat))
                    {
                        if (!SafeValidTarget(potion, target)) continue;
                        var a = d.Add(ActionType.UsePotion, vocab.Id(potion), slot, potion, target,
                            label: $"potion {potion.Id.Entry}{(target != null ? " -> " + TargetLabel(target, player) : "")}");
                        a.Num[1] = 1;
                    }
                }
                if (player.CanRemovePotions)
                    d.Add(ActionType.DiscardPotion, vocab.Id(potion), slot, potion, index: slot, label: $"discard {potion.Id.Entry}");
            }
            slot++;
        }
        d.Add(ActionType.EndTurn, label: "end turn");
    }

    static string TargetLabel(Creature t, Player player) =>
        t.Monster?.Id.Entry ?? (ReferenceEquals(t, player.Creature) ? "self" : "ally " + (t.Player?.Character.Id.Entry ?? t.PetOwner?.Character.Id.Entry ?? "?"));

    /// <summary>Co-op: other players' creatures that can be targeted (alive teammates).</summary>
    static IEnumerable<Creature> Allies(Player player, ICombatState? combat)
    {
        if (combat == null) yield break;
        foreach (var c in combat.PlayerCreatures)
            if (!ReferenceEquals(c, player.Creature) && c.IsAlive && c.IsPlayer) yield return c;
    }

    static bool SafeValidTarget(PotionModel p, Creature? t)
    {
        try { return p.IsValidTarget(t); } catch { return false; }
    }

    public static bool CanUsePotionNow(PotionModel potion, bool inCombat)
    {
        var owner = potion.Owner;
        if (potion.IsQueued || owner == null || owner.Creature.IsDead || !owner.CanRemovePotions) return false;
        if (inCombat && CombatManager.Instance.PlayerActionsDisabled) return false;
        try { if (!potion.PassesCustomUsabilityCheck) return false; } catch { return false; }
        return potion.Usage switch
        {
            PotionUsage.CombatOnly => inCombat,
            PotionUsage.AnyTime => true,
            _ => false,
        };
    }

    /// <summary>
    /// Targets for a card/potion of the given target type. Only AnyEnemy takes an explicit
    /// creature; every other type is played with a null target (the game fills self/all/random).
    /// </summary>
    public static IEnumerable<Creature?> CandidateTargets(TargetType type, Player player, ICombatState? combat)
    {
        if (type == TargetType.AnyEnemy)
        {
            if (combat != null)
                foreach (var e in combat.HittableEnemies) yield return e;
            yield break;
        }
        // co-op: a teammate (AnyAlly) or any player (AnyPlayer) is chosen explicitly
        if (type is TargetType.AnyAlly or TargetType.AnyPlayer && combat != null && combat.PlayerCreatures.Count > 1)
        {
            if (type == TargetType.AnyPlayer) yield return player.Creature;
            foreach (var c in Allies(player, combat)) yield return c;
            yield break;
        }
        yield return null;
    }

    /// <summary>Potions (unlike cards) take an explicit creature for Self / AnyPlayer.</summary>
    public static IEnumerable<Creature?> PotionTargets(TargetType type, Player player, ICombatState? combat)
    {
        switch (type)
        {
            case TargetType.AnyEnemy:
                if (combat != null)
                    foreach (var e in combat.HittableEnemies) yield return e;
                break;
            case TargetType.Self:
                yield return player.Creature;
                break;
            case TargetType.AnyPlayer:
                yield return player.Creature;
                foreach (var c in Allies(player, combat)) yield return c;
                break;
            case TargetType.AnyAlly:
                foreach (var c in Allies(player, combat)) yield return c;
                break;
            default:
                yield return null;
                break;
        }
    }
}

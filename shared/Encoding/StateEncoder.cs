using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using static Sts2Ai.Shared.Encoding.ObsBuilder;

namespace Sts2Ai.Shared.Encoding;

/// <summary>
/// Encodes the parts of the game state common to every decision. Decision-specific code
/// (combat actions, rewards, shop, ...) adds offer tokens and action tokens afterwards and uses
/// the lookup tables here (card -> token, creature -> token) to point actions at entities.
/// Shared verbatim between the headless simulator and the in-game mod.
/// </summary>
public sealed class StateEncoder
{
    public readonly Vocab Vocab;
    public readonly ObsBuilder B = new();
    public readonly Dictionary<CardModel, int> CardTok = new(ReferenceEqualityComparer.Instance);
    public readonly Dictionary<Creature, int> CreatureTok = new(ReferenceEqualityComparer.Instance);
    public readonly Dictionary<PotionModel, int> PotionTok = new(ReferenceEqualityComparer.Instance);
    readonly List<CardModel> _tmpCards = new();

    public StateEncoder(Vocab vocab) => Vocab = vocab;

    /// <summary>Encode co-op teammates as tokens (models trained on co-op runs).</summary>
    public static bool EncodeAllies;

    public void Begin(DecisionType decision)
    {
        B.Clear();
        B.Decision = decision;
        CardTok.Clear();
        CreatureTok.Clear();
        PotionTok.Clear();
    }

    /// <summary>Encode the policy's bounded structured observation, including internal identifiers; see the observation contract.</summary>
    public void EncodeState(IRunState run, Player player, DecisionType decision, RunHistorySummary? history)
    {
        Begin(decision);
        var pcs = player.PlayerCombatState;
        var combat = player.Creature.CombatState;
        bool inCombat = pcs != null && combat != null && CombatManager.Instance.IsInProgress;

        EncodeGlobal(run, player, decision, inCombat);
        EncodeCreature(player.Creature, TokenType.Player, 0, combat);
        int slot = 1;
        // co-op: teammates (and their pets) are player tokens flagged as allies (features 28..31); only for
        // models whose observation contract enables them (EncodeAllies).
        if (EncodeAllies && run.Players.Count > 1)
            foreach (var ally in run.Players)
            {
                if (ReferenceEquals(ally, player)) continue;
                MarkAlly(EncodeCreature(ally.Creature, TokenType.Player, slot++, combat), ally, inCombat);
                if (inCombat && ally.PlayerCombatState != null)
                    foreach (var pet in ally.PlayerCombatState.Pets)
                        if (!CreatureTok.ContainsKey(pet)) MarkAlly(EncodeCreature(pet, TokenType.Player, slot++, combat), ally, inCombat);
            }
        if (inCombat)
        {
            foreach (var e in combat!.Enemies)
            {
                if (!e.IsAlive && !e.IsMonster) continue;
                EncodeCreature(e, TokenType.Enemy, slot++, combat);
            }
            foreach (var pet in player.PlayerCombatState!.Pets)
                if (!CreatureTok.ContainsKey(pet)) EncodeCreature(pet, TokenType.Player, slot++, combat);
            foreach (var (c, s) in CreatureTok.ToList()) EncodePowers(c, B.TokCat[s * ObsSpec.NTokCat + 4]);

            int pos = 0;
            foreach (var card in pcs!.Hand.Cards) EncodeCard(card, TokenType.HandCard, pos++, playableCheck: true);
            foreach (var card in pcs.PlayPile.Cards) EncodeCard(card, TokenType.PlayPile, 0);
            int oi = 0;
            foreach (var orb in pcs.OrbQueue.Orbs) EncodeOrb(orb, oi++, pcs.OrbQueue.Capacity);
        }
        else
        {
            EncodePowers(player.Creature, 0);
        }

        EncodePotions(player, inCombat);
        foreach (var bossTok in EncodeBosses(run)) { }
        EncodeRelics(player);
        history?.Encode(this, run, player);

        if (inCombat)
        {
            EncodePileMultiset(pcs!.DrawPile.Cards, TokenType.DrawCard);
            EncodePileMultiset(pcs.DiscardPile.Cards, TokenType.DiscardCard);
            EncodePileMultiset(pcs.ExhaustPile.Cards, TokenType.ExhaustCard);
        }
        // The master deck matters for every strategic decision; in combat it is mostly redundant with
        // the piles, so it is only added when there is room.
        if (!inCombat || B.TokenCapacityLeft > 60)
            EncodePileMultiset(player.Deck.Cards, TokenType.DeckCard);
    }

    void EncodeGlobal(IRunState run, Player player, DecisionType decision, bool inCombat)
    {
        var rs = run as RunState;
        var pcs = player.PlayerCombatState;
        var cr = player.Creature;
        int t = B.Token(TokenType.Global, Vocab.Id(player.Character), Vocab.Id(run.Act), (int)decision,
            Math.Min(run.CurrentActIndex, 7));
        var n = B.Num(t);
        n[0] = run.CurrentActIndex / 3f;
        n[1] = (rs?.ActFloor ?? 0) / 17f;
        n[2] = (rs?.TotalFloor ?? 0) / 50f;
        n[3] = cr.CurrentHp / 100f;
        n[4] = cr.MaxHp / 100f;
        n[5] = cr.MaxHp > 0 ? cr.CurrentHp / (float)cr.MaxHp : 0;
        n[6] = player.Gold / 300f;
        n[7] = player.MaxEnergy / 5f;
        n[8] = run.AscensionLevel / 10f;
        n[9] = player.Deck.Cards.Count / 30f;
        n[10] = player.MaxPotionCount / 5f;
        n[11] = player.Potions.Count() / 5f;
        n[12] = inCombat ? 1 : 0;
        n[13] = (rs?.CurrentRoom?.RoomType ?? RoomType.Unassigned) switch
        {
            RoomType.Elite => 1, RoomType.Boss => 2, _ => 0,
        } / 2f;
        n[14] = player.Relics.Count / 20f;
        if (inCombat && pcs != null)
        {
            n[15] = pcs.Energy / 5f;
            n[16] = pcs.MaxEnergy / 5f;
            n[17] = pcs.Stars / 5f;
            n[18] = pcs.TurnNumber / 10f;
            n[19] = pcs.Hand.Cards.Count / 10f;
            n[20] = pcs.DrawPile.Cards.Count / 30f;
            n[21] = pcs.DiscardPile.Cards.Count / 30f;
            n[22] = pcs.ExhaustPile.Cards.Count / 20f;
            n[23] = pcs.OrbQueue.Capacity / 5f;
            var combat = cr.CombatState;
            if (combat != null)
            {
                int incoming = 0, enemyHp = 0, alive = 0;
                foreach (var e in combat.Enemies)
                {
                    if (!e.IsAlive) continue;
                    alive++;
                    enemyHp += e.CurrentHp + e.Block;
                    incoming += IncomingDamage(e, combat);
                }
                n[24] = alive / 5f;
                n[25] = enemyHp / 200f;
                n[26] = incoming / 50f;
                n[27] = Math.Max(0, incoming - cr.Block) / 50f;
                n[28] = cr.CurrentHp > 0 ? Math.Min(4f, Math.Max(0, incoming - cr.Block) / (float)cr.CurrentHp) : 0;
            }
        }
        n[29] = (rs?.Map.SecondBossMapPoint != null) ? 1 : 0;
        n[30] = rs?.VisitedMapCoords.Count / 17f ?? 0;
        if (EncodeAllies) n[31] = (run.Players.Count - 1) / 3f; // co-op: number of teammates
    }

    void MarkAlly(int t, Player ally, bool inCombat)
    {
        if (t < 0) return;
        var n = B.Num(t);
        n[28] = 1;
        var pcs = ally.PlayerCombatState;
        if (inCombat && pcs != null)
        {
            n[29] = CombatManager.Instance.IsPlayerReadyToEndTurn(ally) ? 1 : 0;
            n[30] = pcs.Hand.Cards.Count / 10f;
            n[31] = pcs.Energy / 5f;
        }
        else
        {
            n[30] = ally.Gold / 300f;
            n[31] = ally.Deck.Cards.Count / 30f;
        }
    }

    static int IncomingDamage(Creature enemy, ICombatState combat)
    {
        if (enemy.Monster == null) return 0;
        int total = 0;
        try
        {
            foreach (var intent in enemy.Monster.NextMove.Intents)
                if (intent is AttackIntent atk)
                    total += atk.GetTotalDamage(combat.PlayerCreatures, enemy);
        }
        catch { /* intent calculators can depend on transient state; treat as unknown */ }
        return total;
    }

    public int EncodeCreature(Creature c, TokenType type, int slot, ICombatState? combat)
    {
        int moveId = 0, firstIntent = 0;
        var mon = c.Monster;
        if (mon != null && combat != null)
        {
            moveId = Vocab.Hash(mon.Id.Entry + "/" + mon.NextMove.Id);
            var intents = mon.NextMove.Intents;
            if (intents.Count > 0) firstIntent = (int)intents[0].IntentType + 1;
        }
        int t = B.Token(type, mon != null ? Vocab.Id(mon) : Vocab.Id(c.Player?.Character), moveId, firstIntent, slot);
        if (t < 0) return t;
        CreatureTok[c] = t;
        var n = B.Num(t);
        n[0] = c.CurrentHp / 100f;
        n[1] = c.MaxHp / 100f;
        n[2] = c.MaxHp > 0 ? c.CurrentHp / (float)c.MaxHp : 0;
        n[3] = c.Block / 30f;
        n[4] = c.IsAlive ? 1 : 0;
        n[5] = c.IsPet ? 1 : 0;
        n[6] = c.IsStunned ? 1 : 0;
        if (mon != null && combat != null)
        {
            var intents = mon.NextMove.Intents;
            n[7] = intents.Count / 3f;
            int single = 0, repeats = 0, total = 0;
            foreach (var intent in intents)
            {
                int k = (int)intent.IntentType;
                if (k >= 0 && k < 15) n[10 + k] = 1;
                if (intent is AttackIntent atk)
                {
                    try
                    {
                        single = Math.Max(single, atk.GetSingleDamage(combat.PlayerCreatures, c));
                        repeats = Math.Max(repeats, Math.Max(1, atk.Repeats));
                        total += atk.GetTotalDamage(combat.PlayerCreatures, c);
                    }
                    catch { }
                }
            }
            n[8] = single / 20f;
            n[9] = repeats / 5f;
            n[25] = total / 30f;
            n[26] = combat.HittableEnemies.Contains(c) ? 1 : 0;
        }
        n[27] = Log1(c.CurrentHp + c.Block) / 5f;
        return t;
    }

    void EncodePowers(Creature c, int ownerSlot)
    {
        foreach (var p in c.Powers)
        {
            int t = B.Token(TokenType.Power, Vocab.Id(p), 0, (int)p.TypeForCurrentAmount + 1, ownerSlot);
            if (t < 0) return;
            var n = B.Num(t);
            n[0] = p.Amount / 10f;
            n[1] = Log1(p.Amount) / 3f;
            n[2] = p.Amount > 0 ? 1 : p.Amount < 0 ? -1 : 0;
        }
    }

    public int EncodeCard(CardModel card, TokenType type, int position, bool playableCheck = false, int count = 1)
    {
        int aux1 = Math.Min((int)card.Rarity, 7) * 16 + Math.Min((int)card.TargetType, 15);
        int t = B.Token(type, Vocab.Id(card), Vocab.Id((AbstractModel?)card.Enchantment ?? card.Affliction), aux1, Math.Min(position, 127));
        if (t < 0) return t;
        CardTok[card] = t;
        FillCardNum(card, B.Num(t), position, playableCheck, count);
        return t;
    }

    static void FillCardNum(CardModel card, Span<float> n, int position, bool playableCheck, int count)
    {
        try
        {
            n[0] = card.EnergyCost.CostsX ? 0 : card.EnergyCost.GetResolved() / 3f;
            n[1] = card.EnergyCost.CostsX ? 1 : 0;
        }
        catch { n[0] = card.EnergyCost.Canonical / 3f; }
        int star = card.CurrentStarCost;
        n[2] = star >= 0 ? star / 3f : 0;
        n[3] = star >= 0 || card.HasStarCostX ? 1 : 0;
        n[4] = card.CurrentUpgradeLevel;
        if (playableCheck)
        {
            try { n[5] = card.CanPlay() ? 1 : 0; } catch { n[5] = 0; }
        }
        int ty = (int)card.Type;
        if (ty >= 1 && ty <= 6) n[5 + ty] = 1; // 6..11
        foreach (var kw in card.Keywords)
        {
            int k = (int)kw;
            if (k >= 1 && k <= 7) n[11 + k] = 1; // 12..18
        }
        int vi = 0;
        foreach (var kv in card.DynamicVars.OrderBy(k => k.Key, StringComparer.Ordinal))
        {
            if (vi >= 6) break;
            decimal v;
            try { v = kv.Value.PreviewValue; } catch { v = kv.Value.BaseValue; }
            n[19 + vi] = (float)v / 15f; // 19..24
            vi++;
        }
        n[25] = card.Enchantment?.Amount / 5f ?? 0;
        n[26] = count / 3f;
        n[27] = position / 10f;
        n[28] = card.IsClone ? 1 : 0;
        n[29] = card.IsRemovable ? 0 : 1;
    }

    /// <summary>Encode a pile as one token per distinct (card, upgrade, enchantment) with a count.</summary>
    public void EncodePileMultiset(IEnumerable<CardModel> cards, TokenType type)
    {
        var groups = new Dictionary<(ModelId, int, ModelId?), (CardModel first, int count)>();
        foreach (var c in cards)
        {
            var key = (c.Id, c.CurrentUpgradeLevel, c.Enchantment?.Id);
            groups[key] = groups.TryGetValue(key, out var g) ? (g.first, g.count + 1) : (c, 1);
        }
        int pos = 0;
        foreach (var (first, count) in groups.Values.OrderBy(g => g.first.Id.Entry, StringComparer.Ordinal))
        {
            if (B.TokenCapacityLeft <= 4) return;
            EncodeCard(first, type, pos++, count: count);
        }
    }

    void EncodeOrb(OrbModel orb, int index, int capacity)
    {
        int t = B.Token(TokenType.Orb, Vocab.Id(orb), 0, 0, index);
        if (t < 0) return;
        var n = B.Num(t);
        try
        {
            n[0] = (float)orb.PassiveVal / 10f;
            n[1] = (float)orb.EvokeVal / 10f;
        }
        catch { }
        n[2] = index / 10f;
        n[3] = capacity / 10f;
    }

    void EncodePotions(Player player, bool inCombat)
    {
        int slot = 0;
        foreach (var p in player.PotionSlots)
        {
            if (p != null)
            {
                int t = B.Token(TokenType.Potion, Vocab.Id(p), 0, (int)p.TargetType, slot);
                if (t >= 0)
                {
                    PotionTok[p] = t;
                    B.Num(t)[0] = inCombat ? 1 : 0;
                }
            }
            slot++;
        }
    }

    IEnumerable<int> EncodeBosses(IRunState run)
    {
        var act = run.Act;
        int t = B.Token(TokenType.Boss, Vocab.Id(act.BossEncounter), 0, 0, 0);
        if (t >= 0) yield return t;
        if (act.SecondBossEncounter != null)
        {
            t = B.Token(TokenType.Boss, Vocab.Id(act.SecondBossEncounter), 0, 0, 1);
            if (t >= 0) yield return t;
        }
    }

    void EncodeRelics(Player player)
    {
        int i = 0;
        foreach (var r in player.Relics)
        {
            int t = B.Token(TokenType.Relic, Vocab.Id(r), 0, (int)r.Status, Math.Min(i++, 127));
            if (t < 0) return;
            var n = B.Num(t);
            n[0] = r.DisplayAmount / 10f;
            n[1] = r.IsUsedUp ? 1 : 0;
            n[2] = r.ShowCounter ? 1 : 0;
        }
    }
}

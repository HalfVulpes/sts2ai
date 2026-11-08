using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rewards;
using Sts2Ai.Shared.Encoding;

namespace Sts2Ai.Shared.Decisions;

/// <summary>Tokens for things offered to the player outside combat (rewards, shop items, options, map nodes).</summary>
public static class OfferEncoding
{
    public static int OfferCard(StateEncoder enc, CardModel card, int position, int price = -1, bool affordable = true)
    {
        int t = enc.EncodeCard(card, TokenType.OfferCard, position);
        if (t >= 0 && price >= 0)
        {
            var n = enc.B.Num(t);
            n[30] = price / 150f;
            n[31] = affordable ? 1 : 0;
        }
        return t;
    }

    public static int OfferRelic(StateEncoder enc, RelicModel relic, int position, int price = -1, bool affordable = true)
    {
        int t = enc.B.Token(TokenType.OfferRelic, enc.Vocab.Id(relic), 0, (int)relic.Rarity, position);
        if (t >= 0)
        {
            var n = enc.B.Num(t);
            n[0] = price >= 0 ? price / 300f : 0;
            n[1] = affordable ? 1 : 0;
        }
        return t;
    }

    public static int OfferPotion(StateEncoder enc, PotionModel potion, int position, int price = -1, bool affordable = true)
    {
        int t = enc.B.Token(TokenType.OfferPotion, enc.Vocab.Id(potion), 0, (int)potion.Rarity, position);
        if (t >= 0)
        {
            var n = enc.B.Num(t);
            n[0] = price >= 0 ? price / 100f : 0;
            n[1] = affordable ? 1 : 0;
        }
        return t;
    }

    /// <summary>Generic option token (event option, rest site option, reward entry, shop service).</summary>
    public static int Option(StateEncoder enc, int vocab, string key, int flags, int position, params float[] num)
    {
        int t = enc.B.Token(TokenType.Option, vocab, enc.Vocab.Hash(key), flags, position);
        if (t >= 0)
        {
            var n = enc.B.Num(t);
            for (int i = 0; i < num.Length && i < n.Length; i++) n[i] = num[i];
        }
        return t;
    }

    public static int Reward(StateEncoder enc, Reward r, int position)
    {
        switch (r)
        {
            case PotionReward pr when pr.Potion != null:
                return OfferPotion(enc, pr.Potion, position);
            case RelicReward rr when rr.Relic != null:
                return OfferRelic(enc, rr.Relic, position);
            case GoldReward gr:
                return Option(enc, 0, "REWARD/GOLD", 1, position, gr.Amount / 100f);
            default:
                return Option(enc, 0, "REWARD/" + r.GetType().Name, 2, position);
        }
    }

    public static int EventOption(StateEncoder enc, EventModel ev, EventOption o, int position)
    {
        int flags = (o.IsLocked ? 1 : 0) | (o.IsProceed ? 2 : 0) | (o.Relic != null ? 4 : 0);
        int t = enc.B.Token(TokenType.Option, enc.Vocab.Id(ev), enc.Vocab.Hash(o.TextKey), flags, position);
        if (t >= 0 && o.Relic != null)
            enc.B.Num(t)[0] = enc.Vocab.Id(o.Relic) / 4096f; // weak hint; the relic also gets its own token
        if (o.Relic != null) OfferRelic(enc, o.Relic, position);
        return t;
    }

    // ------------------------------------------------------------------ map

    static readonly MapPointType[] Kinds =
    {
        MapPointType.Monster, MapPointType.Elite, MapPointType.Unknown, MapPointType.RestSite,
        MapPointType.Shop, MapPointType.Treasure,
    };

    /// <summary>Per-node min/max counts of each room kind on any path from the node to the act boss.</summary>
    public sealed class PathStats
    {
        readonly Dictionary<MapPoint, (int[] min, int[] max, int depth)> _memo = new(ReferenceEqualityComparer.Instance);

        public (int[] min, int[] max, int depth) Get(MapPoint p)
        {
            if (_memo.TryGetValue(p, out var r)) return r;
            var min = new int[Kinds.Length];
            var max = new int[Kinds.Length];
            int depth = 0;
            if (p.Children.Count > 0)
            {
                Array.Fill(min, int.MaxValue);
                foreach (var c in p.Children)
                {
                    var (cmin, cmax, cd) = Get(c);
                    for (int k = 0; k < Kinds.Length; k++)
                    {
                        min[k] = Math.Min(min[k], cmin[k]);
                        max[k] = Math.Max(max[k], cmax[k]);
                    }
                    depth = Math.Max(depth, cd + 1);
                }
            }
            int self = Array.IndexOf(Kinds, p.PointType);
            if (self >= 0) { min[self]++; max[self]++; }
            r = (min, max, depth);
            _memo[p] = r;
            return r;
        }
    }

    public static int MapNode(StateEncoder enc, MapPoint p, int position, PathStats stats)
    {
        int t = enc.B.Token(TokenType.MapNode, 0, enc.Vocab.Hash("MAP/" + p.PointType), (int)p.PointType, position);
        if (t < 0) return t;
        var n = enc.B.Num(t);
        var (min, max, depth) = stats.Get(p);
        for (int k = 0; k < Kinds.Length; k++)
        {
            n[k] = min[k] / 5f;
            n[6 + k] = max[k] / 5f;
        }
        n[12] = depth / 16f;
        n[13] = p.coord.row / 16f;
        n[14] = p.coord.col / 7f;
        n[15] = p.Children.Count / 3f;
        return t;
    }
}

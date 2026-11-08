using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Models;
using Sts2Ai.Shared.Encoding;

namespace Sts2Ai.Shared.Decisions;

/// <summary>Writes a Decision's legal actions as action tokens, resolving entity pointers.</summary>
public static class ActionEncoder
{
    public static void Encode(StateEncoder enc, Decision d, IReadOnlyDictionary<object, int>? extraRefs = null)
    {
        foreach (var a in d.Actions)
        {
            int r0 = Ref(enc, a.Payload, extraRefs), r1 = Ref(enc, a.Target, extraRefs);
            int i = enc.B.Action(a.Type, a.Vocab, a.Aux, r0, r1);
            if (i < 0) break;
            a.Num.AsSpan().CopyTo(enc.B.ActionNum(i));
        }
        enc.B.Sanitize(); // last step of every decision's encoding
    }

    static int Ref(StateEncoder enc, object? o, IReadOnlyDictionary<object, int>? extra)
    {
        switch (o)
        {
            case null: return -1;
            case CardModel c when enc.CardTok.TryGetValue(c, out var t): return t;
            case Creature cr when enc.CreatureTok.TryGetValue(cr, out var t2): return t2;
            case PotionModel p when enc.PotionTok.TryGetValue(p, out var t3): return t3;
        }
        return extra != null && extra.TryGetValue(o, out var t4) ? t4 : -1;
    }
}

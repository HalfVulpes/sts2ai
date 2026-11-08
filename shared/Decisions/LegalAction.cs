using Sts2Ai.Shared.Encoding;

namespace Sts2Ai.Shared.Decisions;

/// <summary>
/// One legal choice at a decision point. Encoded as an action token (type, vocab id, aux,
/// pointers to entity tokens, numeric features) and carrying the game objects needed to
/// execute it (Payload/Target/Index).
/// </summary>
public sealed class LegalAction
{
    public ActionType Type;
    public int Vocab;
    public int Aux;
    public object? Payload;   // card / potion / reward / option / map point / merchant entry ...
    public object? Target;    // target creature, if any
    public int Index = -1;    // option index etc.
    public readonly float[] Num = new float[ObsSpec.FAct];
    public string Label = "";

    public override string ToString() => Label.Length > 0 ? Label : $"{Type}#{Index}";
}

/// <summary>A pending decision: what kind it is and the full list of legal actions.</summary>
public sealed class Decision
{
    public DecisionType Type;
    public readonly List<LegalAction> Actions = new();
    /// <summary>Minimum / maximum number of picks for multi-select card prompts (handled as sequential picks).</summary>
    public int MinSelect, MaxSelect;
    public string Context = "";

    public LegalAction Add(ActionType type, int vocab = 0, int aux = 0, object? payload = null, object? target = null, int index = -1, string label = "")
    {
        var a = new LegalAction { Type = type, Vocab = vocab, Aux = aux, Payload = payload, Target = target, Index = index, Label = label };
        Actions.Add(a);
        return a;
    }
}

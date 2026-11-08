namespace Sts2Ai.Shared.Encoding;

/// <summary>
/// Accumulates entity tokens and legal-action tokens for one decision into flat arrays
/// laid out exactly as the model expects (see ObsSpec / spec.py). Reused across decisions.
/// </summary>
public sealed class ObsBuilder
{
    public readonly int[] TokCat = new int[ObsSpec.MaxTokens * ObsSpec.NTokCat];
    public readonly float[] TokNum = new float[ObsSpec.MaxTokens * ObsSpec.FTok];
    public readonly int[] ActCat = new int[ObsSpec.MaxActions * ObsSpec.NActCat];
    public readonly int[] ActRef = new int[ObsSpec.MaxActions * 2];
    public readonly float[] ActNum = new float[ObsSpec.MaxActions * ObsSpec.FAct];

    /// <summary>FNV-1a over the encoded observation (tokens and actions in use). Equal hashes mean the mod and
    /// a headless mirror see the same decision (used to check that a search server is in sync).</summary>
    public ulong Hash()
    {
        ulong h = 14695981039346656037UL;
        void Mix(int v) { h = (h ^ (uint)v) * 1099511628211UL; }
        Mix(NumTokens); Mix(NumActions);
        for (int i = 0; i < NumTokens * ObsSpec.NTokCat; i++) Mix(TokCat[i]);
        for (int i = 0; i < NumTokens * ObsSpec.FTok; i++) Mix(BitConverter.SingleToInt32Bits(TokNum[i]));
        for (int i = 0; i < NumActions * ObsSpec.NActCat; i++) Mix(ActCat[i]);
        for (int i = 0; i < NumActions * 2; i++) Mix(ActRef[i]);
        for (int i = 0; i < NumActions * ObsSpec.FAct; i++) Mix(BitConverter.SingleToInt32Bits(ActNum[i]));
        return h;
    }

    public int NumTokens { get; private set; }
    public int NumActions { get; private set; }
    public int DroppedTokens { get; private set; }
    public int DroppedActions { get; private set; }
    public DecisionType Decision { get; set; }

    public void Clear()
    {
        Array.Clear(TokCat, 0, NumTokens * ObsSpec.NTokCat);
        Array.Clear(TokNum, 0, NumTokens * ObsSpec.FTok);
        Array.Clear(ActCat, 0, NumActions * ObsSpec.NActCat);
        Array.Clear(ActNum, 0, NumActions * ObsSpec.FAct);
        Array.Fill(ActRef, -1, 0, ObsSpec.MaxActions * 2);
        NumTokens = NumActions = DroppedTokens = DroppedActions = 0;
        Decision = DecisionType.None;
    }

    public ObsBuilder() => Clear();

    /// <summary>Add a token; returns its index, or -1 if the token budget is exhausted.</summary>
    public int Token(TokenType type, int vocab, int vocab2 = 0, int aux1 = 0, int aux2 = 0)
    {
        if (NumTokens >= ObsSpec.MaxTokens) { DroppedTokens++; return -1; }
        int i = NumTokens++;
        int o = i * ObsSpec.NTokCat;
        TokCat[o] = (int)type;
        TokCat[o + 1] = vocab;
        TokCat[o + 2] = vocab2;
        TokCat[o + 3] = Math.Clamp(aux1, 0, ObsSpec.AuxVocab - 1);
        TokCat[o + 4] = Math.Clamp(aux2, 0, ObsSpec.AuxVocab - 1);
        return i;
    }

    /// <summary>Numeric features of token i (span of length FTok).</summary>
    public Span<float> Num(int token) => token < 0 ? Span<float>.Empty : TokNum.AsSpan(token * ObsSpec.FTok, ObsSpec.FTok);

    public int TokenCapacityLeft => ObsSpec.MaxTokens - NumTokens;

    /// <summary>Add a legal action; returns its index or -1 when full.</summary>
    public int Action(ActionType type, int vocab = 0, int aux = 0, int ref0 = -1, int ref1 = -1)
    {
        if (NumActions >= ObsSpec.MaxActions) { DroppedActions++; return -1; }
        int i = NumActions++;
        ActCat[i * 3] = (int)type;
        ActCat[i * 3 + 1] = vocab;
        ActCat[i * 3 + 2] = Math.Clamp(aux, 0, ObsSpec.AuxVocab - 1);
        ActRef[i * 2] = ref0;
        ActRef[i * 2 + 1] = ref1;
        return i;
    }

    public Span<float> ActionNum(int action) => action < 0 ? Span<float>.Empty : ActNum.AsSpan(action * ObsSpec.FAct, ObsSpec.FAct);

    /// <summary>Largest magnitude a numeric feature may take (the trainer stores features as float16).</summary>
    public const float MaxFeature = 64f;

    /// <summary>
    /// Bound every numeric feature of the current decision: NaN/inf -> 0, clamp to +-MaxFeature.
    /// Some game values are unbounded (power amounts, scaling card values, counters), so a single
    /// extreme state must not be able to produce inf after the float16 cast and poison training.
    /// </summary>
    public void Sanitize()
    {
        Clamp(TokNum.AsSpan(0, NumTokens * ObsSpec.FTok));
        Clamp(ActNum.AsSpan(0, NumActions * ObsSpec.FAct));
    }

    static void Clamp(Span<float> xs)
    {
        for (int i = 0; i < xs.Length; i++)
        {
            float x = xs[i];
            xs[i] = float.IsNaN(x) ? 0f : Math.Clamp(x, -MaxFeature, MaxFeature);
        }
    }

    /// <summary>Squash an unbounded count/amount into a well-scaled feature.</summary>
    public static float Sc(double v, double scale) => (float)(v / scale);
    public static float Log1(double v) => (float)(Math.Sign(v) * Math.Log(1 + Math.Abs(v)));
}

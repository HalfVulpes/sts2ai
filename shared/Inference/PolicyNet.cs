using System.Numerics;
using System.Runtime.InteropServices;
using System.Text.Json;
using Sts2Ai.Shared.Encoding;

namespace Sts2Ai.Shared.Inference;

/// <summary>
/// CPU float32 re-implementation of train/sts2ai/model.py (Sts2Net) for use inside the game mod,
/// with no native dependencies. Weights come from train/sts2ai/export.py ("STS2NET1" file).
/// Equivalence with PyTorch is checked by `Sts2Sim.dll parity` (see export.py).
/// </summary>
public sealed class PolicyNet
{
    public sealed class Output
    {
        public float[] Logits = Array.Empty<float>(); // one per legal action (padding excluded)
        public float[] Probs = Array.Empty<float>();
        public float VReturn, PWin, Floor, Hp;
    }

    readonly Dictionary<string, (int[] shape, float[] data)> _t = new();
    public readonly int D, Layers, Heads, Ff, Memory, VocabSize;

    /// <summary>Trained on co-op runs: expects teammates encoded as ally tokens (StateEncoder.EncodeAllies).</summary>

    public bool EncodeAllies { get; }


    public PolicyNet(string path)
    {
        using var f = File.OpenRead(path);
        using var br = new BinaryReader(f);
        var magic = new string(br.ReadChars(8));
        if (magic != "STS2NET1") throw new InvalidDataException("not an STS2NET1 model file");
        int jlen = br.ReadInt32();
        var meta = JsonDocument.Parse(br.ReadBytes(jlen)).RootElement;
        var cfg = meta.GetProperty("config");
        D = cfg.GetProperty("d_model").GetInt32();
        Layers = cfg.GetProperty("n_layers").GetInt32();
        Heads = cfg.GetProperty("n_heads").GetInt32();
        Ff = cfg.GetProperty("d_ff").GetInt32();
        Memory = cfg.GetProperty("n_memory").GetInt32();
        VocabSize = cfg.GetProperty("vocab_size").GetInt32();
        EncodeAllies = meta.TryGetProperty("encode_allies", out var ea) && ea.GetBoolean();
        long dataStart = f.Position;
        foreach (var t in meta.GetProperty("tensors").EnumerateArray())
        {
            var shape = t.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            long off = t.GetProperty("offset").GetInt64();
            int n = shape.Aggregate(1, (a, b) => a * b);
            var data = new float[n];
            f.Position = dataStart + off * 4;
            var bytes = br.ReadBytes(n * 4);
            MemoryMarshal.Cast<byte, float>(bytes).CopyTo(data);
            _t[t.GetProperty("name").GetString()!] = (shape, data);
        }
    }

    float[] W(string name) => _t.TryGetValue(name, out var t) ? t.data : throw new KeyNotFoundException(name);

    /// <summary>Run the forward pass on the calling thread only (no thread pool). Needed in a fork()ed simulator
    /// child, where the pool's worker threads do not exist.</summary>
    public static bool Sequential;

    static void For(int n, Action<int> body)
    {
        if (Sequential) { for (int i = 0; i < n; i++) body(i); }
        else Parallel.For(0, n, body);
    }

    // ------------------------------------------------------------------ primitives

    /// <summary>y[r, :] = x[r, :] @ W^T + b   (W is [outDim, inDim] as in torch.nn.Linear)</summary>
    static float[] Linear(float[] x, int rows, int inDim, float[] w, float[]? b, int outDim)
    {
        var y = new float[rows * outDim];
        For(rows, r =>
        {
            var xr = new ReadOnlySpan<float>(x, r * inDim, inDim);
            for (int o = 0; o < outDim; o++)
            {
                float s = Dot(xr, new ReadOnlySpan<float>(w, o * inDim, inDim));
                y[r * outDim + o] = b == null ? s : s + b[o];
            }
        });
        return y;
    }

    static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int n = a.Length, i = 0;
        var acc = Vector<float>.Zero;
        int w = Vector<float>.Count;
        for (; i <= n - w; i += w) acc += new Vector<float>(a.Slice(i, w)) * new Vector<float>(b.Slice(i, w));
        float s = Vector.Dot(acc, Vector<float>.One);
        for (; i < n; i++) s += a[i] * b[i];
        return s;
    }

    static void LayerNorm(float[] x, int rows, int d, float[] g, float[] b, float[] y)
    {
        for (int r = 0; r < rows; r++)
        {
            int o = r * d;
            double mean = 0, var = 0;
            for (int i = 0; i < d; i++) mean += x[o + i];
            mean /= d;
            for (int i = 0; i < d; i++) { double t = x[o + i] - mean; var += t * t; }
            var /= d;
            float inv = (float)(1.0 / Math.Sqrt(var + 1e-5));
            for (int i = 0; i < d; i++) y[o + i] = (float)(x[o + i] - mean) * inv * g[i] + b[i];
        }
    }

    /// <summary>Exact (erf) GELU, as torch's default.</summary>
    static void Gelu(float[] x)
    {
        for (int i = 0; i < x.Length; i++) x[i] = (float)(0.5 * x[i] * (1.0 + Erf(x[i] / Math.Sqrt(2.0))));
    }

    /// <summary>erf to ~1e-7 (Numerical Recipes erfc Chebyshev approximation).</summary>
    static double Erf(double x)
    {
        double z = Math.Abs(x), t = 1.0 / (1.0 + 0.5 * z);
        double r = t * Math.Exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806
                   + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
        return x >= 0 ? 1 - r : r - 1;
    }

    static void Add(float[] y, float[] x) { for (int i = 0; i < y.Length; i++) y[i] += x[i]; }

    void Mlp2(string prefix, float[] x, int rows, int inDim, float[] outBuf, int outOffsetStride)
    {
        // Sequential(Linear(in, D), GELU, Linear(D, D)) added into outBuf rows
        var h = Linear(x, rows, inDim, W(prefix + ".0.weight"), W(prefix + ".0.bias"), D);
        Gelu(h);
        var y = Linear(h, rows, D, W(prefix + ".2.weight"), W(prefix + ".2.bias"), D);
        for (int r = 0; r < rows; r++)
            for (int i = 0; i < D; i++) outBuf[r * outOffsetStride + i] += y[r * D + i];
    }

    // ------------------------------------------------------------------ forward

    /// <summary>Run the network on one encoded decision.</summary>
    public Output Forward(ObsBuilder b) =>
        Forward(b.NumTokens, b.TokCat, b.TokNum, b.NumActions, b.ActCat, b.ActRef, b.ActNum);

    public Output Forward(int nt, int[] tokCat, float[] tokNum, int na, int[] actCat, int[] actRef, float[] actNum)
    {
        int d = D, S = nt + Memory;
        int Clamp(int v, int hi) => Math.Clamp(v, 0, hi - 1);

        // ---- token embeddings
        var h = new float[S * d];
        var tokType = W("tok_type.weight"); var vocab = W("vocab.weight"); var vocab2 = W("vocab2.weight");
        var aux1 = W("aux1.weight"); var aux2 = W("aux2.weight");
        int d2 = d / 2;
        var v2 = new float[nt * d2];
        for (int t = 0; t < nt; t++)
        {
            int o = t * ObsSpec.NTokCat;
            int ty = tokCat[o], v = Clamp(tokCat[o + 1], VocabSize), vv = Clamp(tokCat[o + 2], VocabSize);
            int a1 = Clamp(tokCat[o + 3], ObsSpec.AuxVocab), a2 = Clamp(tokCat[o + 4], ObsSpec.AuxVocab);
            for (int i = 0; i < d; i++)
                h[t * d + i] = tokType[ty * d + i] + vocab[v * d + i] + aux1[a1 * d + i] + aux2[a2 * d + i];
            Array.Copy(vocab2, vv * d2, v2, t * d2, d2);
        }
        var v2p = Linear(v2, nt, d2, W("vocab2_proj.weight"), null, d);
        for (int i = 0; i < nt * d; i++) h[i] += v2p[i];
        var tn = new float[nt * ObsSpec.FTok];
        Array.Copy(tokNum, tn, nt * ObsSpec.FTok);
        Mlp2("tok_num", tn, nt, ObsSpec.FTok, h, d);
        // memory tokens
        var mem = W("memory"); var memType = W("mem_type");
        for (int m = 0; m < Memory; m++)
            for (int i = 0; i < d; i++) h[(nt + m) * d + i] = mem[m * d + i] + memType[i];
        // padding mask (tokens of type PAD are masked as keys)
        var keep = new bool[S];
        for (int t = 0; t < nt; t++) keep[t] = tokCat[t * ObsSpec.NTokCat] != (int)TokenType.Pad;
        for (int m = 0; m < Memory; m++) keep[nt + m] = true;

        // ---- transformer blocks
        int hd = d / Heads;
        float scale = 1f / MathF.Sqrt(hd);
        var ln = new float[S * d];
        for (int l = 0; l < Layers; l++)
        {
            string p = $"blocks.{l}.";
            LayerNorm(h, S, d, W(p + "ln1.weight"), W(p + "ln1.bias"), ln);
            var qkv = Linear(ln, S, d, W(p + "qkv.weight"), W(p + "qkv.bias"), 3 * d);
            var att = new float[S * d];
            For(Heads * S, job =>
            {
                int head = job / S, qi = job % S;
                var sc = new float[S];
                float mx = float.NegativeInfinity;
                var q = new ReadOnlySpan<float>(qkv, qi * 3 * d + head * hd, hd);
                for (int k = 0; k < S; k++)
                {
                    if (!keep[k]) { sc[k] = float.NegativeInfinity; continue; }
                    sc[k] = Dot(q, new ReadOnlySpan<float>(qkv, k * 3 * d + d + head * hd, hd)) * scale;
                    if (sc[k] > mx) mx = sc[k];
                }
                double sum = 0;
                for (int k = 0; k < S; k++) { sc[k] = keep[k] ? MathF.Exp(sc[k] - mx) : 0f; sum += sc[k]; }
                int o = qi * d + head * hd;
                for (int k = 0; k < S; k++)
                {
                    if (sc[k] == 0f) continue;
                    float w = (float)(sc[k] / sum);
                    int vo = k * 3 * d + 2 * d + head * hd;
                    for (int i = 0; i < hd; i++) att[o + i] += w * qkv[vo + i];
                }
            });
            Add(h, Linear(att, S, d, W(p + "proj.weight"), W(p + "proj.bias"), d));
            LayerNorm(h, S, d, W(p + "ln2.weight"), W(p + "ln2.bias"), ln);
            var f1 = Linear(ln, S, d, W(p + "ff1.weight"), W(p + "ff1.bias"), Ff);
            Gelu(f1);
            Add(h, Linear(f1, S, Ff, W(p + "ff2.weight"), W(p + "ff2.bias"), d));
        }
        LayerNorm(h, S, d, W("final_norm.weight"), W("final_norm.bias"), ln);
        h = ln;

        // ---- summary: GLOBAL token + attention pooling over live tokens
        var poolQ = W("pool_q");
        var ps = new double[S];
        double pmx = double.NegativeInfinity;
        for (int t = 0; t < S; t++)
        {
            if (!keep[t]) { ps[t] = double.NegativeInfinity; continue; }
            ps[t] = Dot(new ReadOnlySpan<float>(h, t * d, d), poolQ) / Math.Sqrt(d);
            pmx = Math.Max(pmx, ps[t]);
        }
        double psum = 0;
        for (int t = 0; t < S; t++) { ps[t] = keep[t] ? Math.Exp(ps[t] - pmx) : 0; psum += ps[t]; }
        var summary = new float[2 * d];
        Array.Copy(h, 0, summary, 0, d);
        for (int t = 0; t < S; t++)
            if (ps[t] > 0)
                for (int i = 0; i < d; i++) summary[d + i] += (float)(ps[t] / psum) * h[t * d + i];

        // ---- actions
        var a = new float[na * d];
        var actType = W("act_type.weight"); var actAux = W("act_aux.weight");
        for (int k = 0; k < na; k++)
        {
            int ty = actCat[k * 3], v = Clamp(actCat[k * 3 + 1], VocabSize), ax = Clamp(actCat[k * 3 + 2], ObsSpec.AuxVocab);
            for (int i = 0; i < d; i++) a[k * d + i] = actType[ty * d + i] + vocab[v * d + i] + actAux[ax * d + i];
        }
        var an = new float[na * ObsSpec.FAct];
        Array.Copy(actNum, an, na * ObsSpec.FAct);
        Mlp2("act_num", an, na, ObsSpec.FAct, a, d);
        var refs = new float[na * 2 * d];
        for (int k = 0; k < na; k++)
            for (int j = 0; j < 2; j++)
            {
                int r = actRef[k * 2 + j];
                if (r >= 0 && r < nt) Array.Copy(h, r * d, refs, (k * 2 + j) * d, d);
            }
        Add(a, Linear(refs, na, 2 * d, W("ref_proj.weight"), W("ref_proj.bias"), d));
        var pin = new float[na * 3 * d];
        for (int k = 0; k < na; k++)
        {
            Array.Copy(a, k * d, pin, k * 3 * d, d);
            Array.Copy(summary, 0, pin, k * 3 * d + d, 2 * d);
        }
        var p1 = Linear(pin, na, 3 * d, W("policy_mlp.0.weight"), W("policy_mlp.0.bias"), d); Gelu(p1);
        var p2 = Linear(p1, na, d, W("policy_mlp.2.weight"), W("policy_mlp.2.bias"), d); Gelu(p2);
        var logits = Linear(p2, na, d, W("policy_mlp.4.weight"), W("policy_mlp.4.bias"), 1);
        for (int k = 0; k < na; k++) if (actCat[k * 3] == (int)ActionType.Pad) logits[k] = float.NegativeInfinity;

        // ---- values
        var vt = Linear(summary, 1, 2 * d, W("value_trunk.0.weight"), W("value_trunk.0.bias"), d); Gelu(vt);
        float Head(string n) => Linear(vt, 1, d, W(n + ".weight"), W(n + ".bias"), 1)[0];

        var res = new Output { Logits = logits, VReturn = Head("v_return"), PWin = 1f / (1f + MathF.Exp(-Head("v_win"))), Floor = Head("v_floor"), Hp = Head("v_hp") };
        float lmx = logits.Length > 0 ? logits.Max() : 0;
        var e = logits.Select(x => float.IsNegativeInfinity(x) ? 0 : MathF.Exp(x - lmx)).ToArray();
        float es = e.Sum();
        res.Probs = e.Select(x => es > 0 ? x / es : 0).ToArray();
        return res;
    }
}

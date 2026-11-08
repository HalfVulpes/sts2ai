using System.Text;
using HarmonyLib;
using MegaCrit.Sts2.Core.Map;

namespace Sts2Sim.Engine;

/// <summary>
/// Exact, faster replacement for <c>MapPathPruning.FindMatchingSegments</c> (map generation spends
/// most of its time there; it was ~60% of simulator CPU).
///
/// The original enumerates every start→boss path (exponentially many) and, for every segment
/// occurrence on every path, builds a string key and does a sorted-dictionary insert. Repeated
/// occurrences of an identical segment never change the result: the copy always "overlaps" the stored
/// one (same interior points), so it is skipped. Only the *first occurrence* of each distinct segment
/// matters, and its order. A segment's first occurrence lies on the lexicographically smallest path
/// (in child-enumeration order) through it, i.e. minPrefix(start→first) + segment + minSuffix(last→boss),
/// at position i = |minPrefix|. So we enumerate distinct segments over the DAG, sort them by
/// (that path, i, length) and replay exactly the original insertions — same keys, same overlap rule,
/// same insertion order, hence identical output (and identical RNG use downstream).
/// <see cref="Verify"/> runs both and compares.
/// </summary>
public static class FastMapPruning
{
    public static bool Verify;
    public static int Verified;
    [ThreadStatic] static bool _bypass;

    public static void Apply(Harmony h) =>
        h.Patch(AccessTools.Method(typeof(MapPathPruning), nameof(MapPathPruning.FindMatchingSegments)),
            prefix: new HarmonyMethod(typeof(FastMapPruning), nameof(Prefix)));

    static bool Prefix(MapPoint startingMapPoint, ref List<List<MapPoint[]>> __result)
    {
        if (_bypass) return true;
        __result = FindMatchingSegments(startingMapPoint);
        if (Verify)
        {
            _bypass = true;
            List<List<MapPoint[]>> orig;
            try { orig = MapPathPruning.FindMatchingSegments(startingMapPoint); }
            finally { _bypass = false; }
            if (!Same(orig, __result)) throw new InvalidOperationException("FastMapPruning mismatch");
            Verified++;
        }
        return false;
    }

    static bool Same(List<List<MapPoint[]>> a, List<List<MapPoint[]>> b)
    {
        if (a.Count != b.Count) return false;
        for (int k = 0; k < a.Count; k++)
        {
            if (a[k].Count != b[k].Count) return false;
            for (int m = 0; m < a[k].Count; m++)
                if (!a[k][m].SequenceEqual(b[k][m])) return false;
        }
        return true;
    }

    sealed class Node
    {
        public MapPoint P = null!;
        public Node[] Kids = Array.Empty<Node>();   // children that can reach a boss, in enumeration order
        public int[] KidIdx = Array.Empty<int>();   // their index in the original Children enumeration
        public bool ReachesBoss;
        public int[]? MinPrefix;                    // child indices of the smallest path start→this
        public int[] MinSuffix = Array.Empty<int>();// child indices of the smallest path this→boss
    }

    public static List<List<MapPoint[]>> FindMatchingSegments(MapPoint start)
    {
        // --- graph restricted to points on some start→boss path (FindAllPaths stops at the first Boss)
        var nodes = new Dictionary<MapPoint, Node>(ReferenceEqualityComparer.Instance);
        var topo = new List<Node>();  // post-order (children before parents)
        Node Build(MapPoint p)
        {
            if (nodes.TryGetValue(p, out var n)) return n;
            n = new Node { P = p };
            nodes[p] = n;
            if (p.PointType == MapPointType.Boss) n.ReachesBoss = true;
            else
            {
                var kids = new List<Node>();
                var idx = new List<int>();
                int i = 0;
                foreach (var c in p.Children)
                {
                    var cn = Build(c);
                    if (cn.ReachesBoss) { kids.Add(cn); idx.Add(i); }
                    i++;
                }
                n.Kids = kids.ToArray();
                n.KidIdx = idx.ToArray();
                n.ReachesBoss = kids.Count > 0;
            }
            topo.Add(n);
            return n;
        }
        var root = Build(start);
        var segments = new SortedDictionary<string, List<MapPoint[]>>(StringComparer.Ordinal);
        if (!root.ReachesBoss) return new List<List<MapPoint[]>>();

        // min suffix: first reachable child, repeatedly (lexicographic minimum of index sequences)
        foreach (var n in topo)
            if (n.Kids.Length > 0)
                n.MinSuffix = Prepend(n.KidIdx[0], n.Kids[0].MinSuffix);
        // min prefix: process parents before children (reverse post-order)
        root.MinPrefix = Array.Empty<int>();
        for (int t = topo.Count - 1; t >= 0; t--)
        {
            var n = topo[t];
            if (n.MinPrefix == null) continue;
            for (int k = 0; k < n.Kids.Length; k++)
            {
                var cand = Append(n.MinPrefix, n.KidIdx[k]);
                var c = n.Kids[k];
                if (c.MinPrefix == null || Compare(cand, c.MinPrefix) < 0) c.MinPrefix = cand;
            }
        }

        // --- distinct segments: from each valid start (not a boss), every route of length >= 3 to a valid end
        var found = new List<(int[] order, int i, int len, MapPoint[] seg)>();
        var pts = new List<MapPoint>();
        var idxs = new List<int>();
        foreach (var s in topo)
        {
            if (s.MinPrefix == null || s.Kids.Length == 0 || !IsValidSegmentStart(s.P)) continue;
            pts.Clear(); idxs.Clear();
            pts.Add(s.P);
            Walk(s);
            void Walk(Node n)
            {
                for (int k = 0; k < n.Kids.Length; k++)
                {
                    var c = n.Kids[k];
                    pts.Add(c.P); idxs.Add(n.KidIdx[k]);
                    if (pts.Count >= 3 && c.P.parents.Count >= 2)
                    {
                        var order = new int[s.MinPrefix!.Length + idxs.Count + c.MinSuffix.Length];
                        s.MinPrefix.CopyTo(order, 0);
                        idxs.CopyTo(order, s.MinPrefix.Length);
                        c.MinSuffix.CopyTo(order, s.MinPrefix.Length + idxs.Count);
                        found.Add((order, s.MinPrefix.Length, pts.Count, pts.ToArray()));
                    }
                    if (c.Kids.Length > 0) Walk(c);
                    pts.RemoveAt(pts.Count - 1); idxs.RemoveAt(idxs.Count - 1);
                }
            }
        }
        found.Sort((x, y) =>
        {
            int c = Compare(x.order, y.order);
            if (c != 0) return c;
            c = x.i.CompareTo(y.i);
            return c != 0 ? c : x.len.CompareTo(y.len);
        });

        // --- replay the original insertions
        var sb = new StringBuilder();
        foreach (var f in found)
        {
            string key = SegmentKey(f.seg, sb);
            if (!segments.TryGetValue(key, out var list)) segments[key] = new List<MapPoint[]> { f.seg };
            else if (!list.Any(e => Overlapping(e, f.seg))) list.Add(f.seg);
        }
        return segments.Values.Where(l => l.Count > 1).ToList();
    }

    static bool IsValidSegmentStart(MapPoint p) => p.Children.Count > 1 || p.coord.row == 0;

    // identical to MapPathPruning.OverlappingSegment
    static bool Overlapping(MapPoint[] a, MapPoint[] b)
    {
        if (a.Length < 3 || b.Length < 3) return false;
        for (int i = 1; i <= a.Length - 2; i++)
            if (Equals(a[i], b[i])) return true;
        return false;
    }

    // identical to MapPathPruning.GenerateSegmentKey
    static string SegmentKey(MapPoint[] seg, StringBuilder sb)
    {
        sb.Clear();
        var a = seg[0];
        var b = seg[^1];
        if (a.coord.row == 0) sb.Append(a.coord.row).Append('-').Append(b.coord.col).Append(',').Append(b.coord.row).Append('-');
        else sb.Append(a.coord.col).Append(',').Append(a.coord.row).Append('-').Append(b.coord.col).Append(',').Append(b.coord.row).Append('-');
        for (int i = 0; i < seg.Length; i++)
        {
            if (i > 0) sb.Append(',');
            sb.Append((int)seg[i].PointType);
        }
        return sb.ToString();
    }

    static int Compare(int[] a, int[] b)
    {
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; i++)
            if (a[i] != b[i]) return a[i].CompareTo(b[i]);
        return a.Length.CompareTo(b.Length);
    }

    static int[] Prepend(int x, int[] a) { var r = new int[a.Length + 1]; r[0] = x; a.CopyTo(r, 1); return r; }
    static int[] Append(int[] a, int x) { var r = new int[a.Length + 1]; a.CopyTo(r, 0); r[^1] = x; return r; }
}

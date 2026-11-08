using System.Reflection;
using System.Text;
using System.Text.Json;
using MegaCrit.Sts2.Core.Models;

namespace Sts2Ai.Shared.Encoding;

/// <summary>
/// Maps game model ids ("CARD.BASH", "MONSTER.FUZZY_WURM_CRAWLER", ...) to embedding indices.
///   0          = padding / none
///   1          = unknown id (new content not seen in training)
///   2 .. N+1   = known ids (sorted, frozen into vocab.json next to the model)
///   N+2 ..     = HashBuckets slots for free-form strings (monster move ids, event option keys)
/// The trained model ships its vocab.json, so the mod maps ids exactly as during training even if
/// the game later adds content.
/// </summary>
public sealed class Vocab
{
    public const int Pad = 0;
    public const int Unknown = 1;

    readonly Dictionary<string, int> _ids;
    public IReadOnlyList<string> Names { get; }
    public int HashBase { get; }
    public int Size => HashBase + ObsSpec.HashBuckets;

    Vocab(IReadOnlyList<string> names)
    {
        Names = names;
        _ids = new Dictionary<string, int>(names.Count);
        for (int i = 0; i < names.Count; i++) _ids[names[i]] = i + 2;
        HashBase = names.Count + 2;
    }

    public int Id(ModelId? id) => id == null ? Pad : _ids.GetValueOrDefault(id.ToString(), Unknown);
    public int Id(AbstractModel? m) => m == null ? Pad : Id(m.Id);
    public int Id(string key) => _ids.GetValueOrDefault(key, Unknown);

    /// <summary>Stable FNV-1a hash of a string into the hash-bucket region.</summary>
    public int Hash(string? s)
    {
        if (string.IsNullOrEmpty(s)) return Pad;
        uint h = 2166136261;
        foreach (byte b in System.Text.Encoding.UTF8.GetBytes(s)) { h ^= b; h *= 16777619; }
        return HashBase + (int)(h % ObsSpec.HashBuckets);
    }

    /// <summary>Build from the live ModelDb (simulator) — every registered model id, sorted.</summary>
    public static Vocab FromModelDb()
    {
        var field = typeof(ModelDb).GetField("_contentById", BindingFlags.Static | BindingFlags.NonPublic)!;
        var dict = (System.Collections.IDictionary)field.GetValue(null)!;
        var names = new List<string>();
        foreach (var k in dict.Keys) names.Add(k.ToString()!);
        names.Sort(StringComparer.Ordinal);
        return new Vocab(names);
    }

    public static Vocab Load(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var names = doc.RootElement.GetProperty("ids").EnumerateArray().Select(e => e.GetString()!).ToList();
        return new Vocab(names);
    }

    public void Save(string path, string gameVersion)
    {
        var obj = new Dictionary<string, object>
        {
            ["obs_version"] = ObsSpec.Version,
            ["game_version"] = gameVersion,
            ["hash_buckets"] = ObsSpec.HashBuckets,
            ["size"] = Size,
            ["ids"] = Names,
        };
        File.WriteAllText(path, JsonSerializer.Serialize(obj, new JsonSerializerOptions { WriteIndented = false }));
    }
}

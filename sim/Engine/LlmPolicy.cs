using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Runs;
using Sts2Ai.Shared.Decisions;
using Sts2Ai.Shared.Encoding;

namespace Sts2Sim.Engine;

/// <summary>
/// A general LLM as the policy (benchmark): every decision is rendered as text (what a human player would
/// see: run status, relics, potions, combat state with intents and full card text, the deck, and the numbered
/// legal actions with their descriptions) and sent to an Anthropic-compatible /v1/messages endpoint. The model
/// answers "ACTION: n". Unparseable or out-of-range answers fall back to action 0 and are counted.
/// </summary>
public sealed class LlmPolicy
{
    readonly HttpClient _http = new() { Timeout = TimeSpan.FromSeconds(180) };
    readonly string _url, _model;
    readonly int _maxTokens;
    public int Calls, Invalid, Retries, InTokens, OutTokens;
    public double Seconds;
    public Action<string>? Log;

    public LlmPolicy(string endpoint, string model, string keyFile, int maxTokens = 400)
    {
        _url = endpoint.TrimEnd('/') + "/messages";
        _model = model;
        _maxTokens = maxTokens;
        _http.DefaultRequestHeaders.Add("x-api-key", File.ReadAllText(keyFile).Trim());
        _http.DefaultRequestHeaders.Add("anthropic-version", "2023-06-01");
    }

    const string System = """
        You are playing Slay the Spire 2, a roguelike deck-building game, and your goal is to win the whole run
        (defeat the final boss of act 3). Each message shows the current state and a numbered list of the legal
        actions. In combat you play cards with energy, then end your turn; enemies then act as their intents show.
        Block absorbs damage until your next turn. Think briefly (at most three sentences), then finish with a
        line of the form:
        ACTION: <number>
        """;

    public int Choose(RunDriver dv, Decision d)
    {
        var prompt = Render(dv, d);
        for (int attempt = 0; attempt < 3; attempt++)
        {
            try
            {
                var t0 = DateTime.UtcNow;
                var body = JsonSerializer.Serialize(new
                {
                    model = _model,
                    max_tokens = _maxTokens,
                    system = System,
                    messages = new[] { new { role = "user", content = prompt } },
                });
                using var req = new HttpRequestMessage(HttpMethod.Post, _url) { Content = new StringContent(body, Encoding.UTF8, "application/json") };
                using var resp = _http.Send(req);
                var json = resp.Content.ReadAsStringAsync().GetAwaiter().GetResult();
                Seconds += (DateTime.UtcNow - t0).TotalSeconds;
                if (!resp.IsSuccessStatusCode) throw new HttpRequestException($"{(int)resp.StatusCode}: {json[..Math.Min(200, json.Length)]}");
                using var doc = JsonDocument.Parse(json);
                var root = doc.RootElement;
                var text = new StringBuilder();
                foreach (var c in root.GetProperty("content").EnumerateArray())
                    if (c.TryGetProperty("text", out var tx)) text.Append(tx.GetString());
                if (root.TryGetProperty("usage", out var u))
                {
                    InTokens += u.TryGetProperty("input_tokens", out var i) ? i.GetInt32() : 0;
                    OutTokens += u.TryGetProperty("output_tokens", out var o) ? o.GetInt32() : 0;
                }
                Calls++;
                var answer = text.ToString();
                var m = Regex.Matches(answer, @"ACTION:\s*\**\s*(\d+)", RegexOptions.IgnoreCase);
                int a = m.Count > 0 ? int.Parse(m[^1].Groups[1].Value) : -1;
                Log?.Invoke($"--- {d.Type}\n{prompt}\n>>> {answer.Trim()}");
                if (a < 0 || a >= d.Actions.Count) { Invalid++; return 0; }
                return a;
            }
            catch (Exception e)
            {
                Retries++;
                Log?.Invoke("request failed: " + e.Message);
                Thread.Sleep(2000 * (attempt + 1));
            }
        }
        Invalid++;
        return 0;
    }

    // ------------------------------------------------------------------ rendering

    static string Clean(string s) => Regex.Replace(Regex.Replace(s ?? "", @"\[/?[a-zA-Z_]+[^\]]*\]", ""), @"\s+", " ").Trim();

    static string Safe(Func<string> f, string fallback)
    {
        try { var s = Clean(f()); return string.IsNullOrWhiteSpace(s) || s.Contains(".title") || s.Contains(".description") ? fallback : s; }
        catch { return fallback; }
    }

    static string Pretty(string id) => string.Join(' ', id.Split('_').Select(w => w.Length == 0 ? w : w[0] + w[1..].ToLowerInvariant()));

    public static string CardText(CardModel c, Creature? target = null)
    {
        string title = Safe(() => c.Title, Pretty(c.Id.Entry) + (c.IsUpgraded ? "+" : ""));
        string cost;
        try { cost = c.EnergyCost.CostsX ? "X" : c.EnergyCost.GetResolved().ToString(); } catch { cost = "?"; }
        string desc = Safe(() => c.GetDescriptionForPile(c.Pile?.Type ?? MegaCrit.Sts2.Core.Entities.Cards.PileType.Deck, target), "");
        string star = c.CurrentStarCost > 0 ? $", {c.CurrentStarCost} star" : "";
        return $"{title} [{cost} energy{star}, {c.Type}] {desc}";
    }

    static string CreatureName(Creature c) =>
        c.Monster != null ? Safe(() => c.Monster.Title.GetFormattedText(), Pretty(c.Monster.Id.Entry))
        : c.Player != null ? Safe(() => c.Player.Character.Title.GetFormattedText(), Pretty(c.Player.Character.Id.Entry)) : "?";

    static string Powers(Creature c)
    {
        var ps = c.Powers.Select(p => $"{Safe(() => p.Title.GetFormattedText(), Pretty(p.Id.Entry))} {p.Amount}").ToList();
        return ps.Count == 0 ? "" : " powers: " + string.Join(", ", ps);
    }

    static string Intent(Creature e, ICombatState combat)
    {
        if (e.Monster == null) return "";
        var parts = new List<string>();
        try
        {
            foreach (var it in e.Monster.NextMove.Intents)
            {
                if (it is AttackIntent atk)
                {
                    int dmg = atk.GetSingleDamage(combat.PlayerCreatures, e);
                    int rep = Math.Max(1, atk.Repeats);
                    parts.Add(rep > 1 ? $"attack {dmg}x{rep}" : $"attack {dmg}");
                }
                else parts.Add(it.IntentType.ToString().ToLowerInvariant());
            }
        }
        catch { }
        return parts.Count == 0 ? "unknown" : string.Join(" + ", parts);
    }

    public static string Render(RunDriver dv, Decision d)
    {
        var st = dv.State;
        var p = dv.Player;
        var sb = new StringBuilder();
        EventModel? visibleEvent = d.Type == DecisionType.Event
            && st.CurrentRoom is MegaCrit.Sts2.Core.Rooms.EventRoom er ? er.LocalMutableEvent : null;
        if (visibleEvent != null)
            foreach (var option in visibleEvent.CurrentOptions)
            {
                // The real NEventOptionButton does this before rendering. Headless has no button.
                visibleEvent.DynamicVars.AddTo(option.Title);
                visibleEvent.DynamicVars.AddTo(option.Description);
            }
        string ch = Safe(() => p.Character.Title.GetFormattedText(), Pretty(p.Character.Id.Entry));
        sb.AppendLine($"Character: {ch}. Act {st.CurrentActIndex + 1} of 3, floor {st.TotalFloor}. HP {p.Creature.CurrentHp}/{p.Creature.MaxHp}. Gold {p.Gold}.");
        sb.AppendLine($"Act boss: {Pretty(st.Act.BossEncounter.Id.Entry)}");
        if (p.Relics.Count > 0)
            sb.AppendLine("Relics: " + string.Join("; ", p.Relics.Select(r => $"{Safe(() => r.Title.GetFormattedText(), Pretty(r.Id.Entry))} ({Safe(() => r.DynamicDescription.GetFormattedText(), "")})")));
        var pots = p.PotionSlots.Where(x => x != null).Select(x => $"{Safe(() => x!.Title.GetFormattedText(), Pretty(x!.Id.Entry))} ({Safe(() => x!.DynamicDescription.GetFormattedText(), "")})").ToList();
        sb.AppendLine($"Potions ({pots.Count}/{p.MaxPotionCount}): " + (pots.Count == 0 ? "none" : string.Join("; ", pots)));
        var combat = p.Creature.CombatState;
        if (CombatManager.Instance.IsInProgress && combat != null && p.PlayerCombatState is { } pcs)
        {
            sb.AppendLine($"COMBAT, turn {pcs.TurnNumber}. Energy {pcs.Energy}/{pcs.MaxEnergy}" + (pcs.Stars > 0 ? $", stars {pcs.Stars}" : "") +
                          $". Your block {p.Creature.Block}.{Powers(p.Creature)}");
            foreach (var orb in pcs.OrbQueue.Orbs) sb.AppendLine($"  orb: {Pretty(orb.Id.Entry)}");
            foreach (var pet in pcs.Pets) sb.AppendLine($"  ally {CreatureName(pet)} HP {pet.CurrentHp}/{pet.MaxHp} block {pet.Block}.{Powers(pet)}");
            sb.AppendLine("Enemies:");
            foreach (var (e, index) in combat.Enemies.Where(x => x.IsAlive).Select((e, i) => (e, i)))
                sb.AppendLine($"  - enemy[{index}] {CreatureName(e)}: HP {e.CurrentHp}/{e.MaxHp}, block {e.Block}, intends: {Intent(e, combat)}.{Powers(e)}");
            sb.AppendLine("Hand:");
            foreach (var c in pcs.Hand.Cards) sb.AppendLine("  - " + CardText(c));
            sb.AppendLine($"Draw pile {pcs.DrawPile.Cards.Count} cards, discard pile {pcs.DiscardPile.Cards.Count}, exhausted {pcs.ExhaustPile.Cards.Count}.");
        }
        else
        {
            var deck = p.Deck.Cards.GroupBy(c => Safe(() => c.Title, Pretty(c.Id.Entry))).Select(g => g.Count() > 1 ? $"{g.Key} x{g.Count()}" : g.Key);
            sb.AppendLine($"Deck ({p.Deck.Cards.Count}): " + string.Join(", ", deck));
        }
        sb.AppendLine();
        if (visibleEvent != null)
        {
            var description = visibleEvent.Description ?? visibleEvent.InitialDescription;
            visibleEvent.DynamicVars.AddTo(description);
            sb.AppendLine("Event: " + Safe(() => visibleEvent.Title.GetFormattedText(), Pretty(visibleEvent.Id.Entry)));
            sb.AppendLine(Safe(() => description.GetFormattedText(), "[Event description unavailable]"));
        }
        sb.AppendLine($"Decision: {d.Type}" + (d.Type == DecisionType.CardSelect ? $" (select {d.MinSelect}-{d.MaxSelect} cards; pick one at a time, confirm when done)" : ""));
        sb.AppendLine("Actions:");
        for (int i = 0; i < d.Actions.Count; i++) sb.AppendLine($"  {i}) {Describe(d.Actions[i])}");
        return sb.ToString();
    }

    /// <summary>Interactive text view. No draw order, RNG state, future rewards or rollout scores.</summary>
    public static string RenderPlayerView(RunDriver dv, Decision d)
    {
        var sb = new StringBuilder(Render(dv, d));
        var st = dv.State;
        var p = dv.Player;
        sb.AppendLine($"Ascension: {st.AscensionLevel}. Map position: {st.CurrentMapPoint?.coord.ToString() ?? "start"}.");
        void Pile(string name, IEnumerable<CardModel> cards)
        {
            sb.AppendLine(name + " (unordered):");
            foreach (var group in cards.Select(c => CardText(c)).GroupBy(x => x).OrderBy(g => g.Key, StringComparer.Ordinal))
                sb.AppendLine($"  {group.Count()}x {group.Key}");
        }
        if (CombatManager.Instance.IsInProgress && p.PlayerCombatState is { } pcs)
        {
            Pile("Draw pile", pcs.DrawPile.Cards);
            Pile("Discard pile", pcs.DiscardPile.Cards);
            Pile("Exhaust pile", pcs.ExhaustPile.Cards);
            sb.AppendLine("Visible status rules:");
            var creatures = new[] { p.Creature }.Concat(p.Creature.CombatState!.Enemies.Where(e => e.IsAlive));
            foreach (var creature in creatures)
                foreach (var power in creature.Powers)
                    sb.AppendLine($"  {CreatureName(creature)} / {Pretty(power.Id.Entry)} {power.Amount}: " +
                        Safe(() => string.Join("; ", power.HoverTips.OfType<MegaCrit.Sts2.Core.HoverTips.HoverTip>()
                            .Select(t => t.Description)), "Tooltip unavailable"));
        }
        else
        {
            Pile("Deck details", p.Deck.Cards);
            if (d.Type == DecisionType.Map)
            {
                sb.AppendLine("Current-act map (public node types and connections; no future encounter identities):");
                foreach (var point in st.Map.GetAllMapPoints().OrderBy(m => m.coord.row).ThenBy(m => m.coord.col))
                    sb.AppendLine($"  ({point.coord.col},{point.coord.row}) {point.PointType} -> " +
                        string.Join(", ", point.Children.OrderBy(m => m.coord.col).Select(m => $"({m.coord.col},{m.coord.row})")));
            }
        }
        return sb.ToString();
    }

    static string Describe(LegalAction a)
    {
        var target = a.Target as Creature;
        string targetIndex = target?.Monster != null
            ? $"enemy[{target.CombatState!.Enemies.Where(e => e.IsAlive).ToList().IndexOf(target)}] " : "";
        string on = target != null ? $" -> target {targetIndex}{CreatureName(target)} (HP {target.CurrentHp}/{target.MaxHp}, block {target.Block})" : "";
        switch (a.Payload)
        {
            case CardModel c when a.Type == ActionType.PlayCard: return "Play " + CardText(c, target) + on;
            case CardModel c when a.Type == ActionType.SelectCard: return "Choose card: " + CardText(c);
            case PotionModel pot when a.Type == ActionType.UsePotion: return $"Use potion {Safe(() => pot.Title.GetFormattedText(), Pretty(pot.Id.Entry))}{on}";
            case PotionModel pot when a.Type == ActionType.DiscardPotion: return $"Discard potion {Safe(() => pot.Title.GetFormattedText(), Pretty(pot.Id.Entry))}";
            case MapPoint mp: return $"Travel to {mp.PointType} (column {mp.coord.col}, row {mp.coord.row}); next: " +
                                     string.Join(", ", mp.Children.Select(ch => ch.PointType.ToString()));
            case EventOption eo: return "Event option: " + Safe(() => eo.Title.GetFormattedText(), eo.TextKey) + " — " + Safe(() => eo.Description.GetFormattedText(), "[Description unavailable]")
                + " " + Safe(() => string.Join("; ", eo.HoverTips.OfType<MegaCrit.Sts2.Core.HoverTips.HoverTip>().Select(t => t.Description)), "");
            case RestSiteOption ro: return "Rest site: " + Safe(() => ro.Title.GetFormattedText(), ro.OptionId) + " — " + Safe(() => ro.Description.GetFormattedText(), "");
            case MerchantCardEntry ce when ce.CreationResult?.Card is { } card: return $"Buy card ({ce.Cost} gold): " + CardText(card);
            case MerchantRelicEntry re when re.Model is { } relic: return $"Buy relic ({re.Cost} gold): {Safe(() => relic.Title.GetFormattedText(), Pretty(relic.Id.Entry))} ({Safe(() => relic.DynamicDescription.GetFormattedText(), "")})";
            case MerchantPotionEntry pe when pe.Model is { } potion: return $"Buy potion ({pe.Cost} gold): {Safe(() => potion.Title.GetFormattedText(), Pretty(potion.Id.Entry))} ({Safe(() => potion.DynamicDescription.GetFormattedText(), "")})";
            case MerchantCardRemovalEntry rem: return $"Pay {rem.Cost} gold to remove a card from your deck";
            case RelicModel r when a.Type == ActionType.SelectRelic: return $"Take relic {Safe(() => r.Title.GetFormattedText(), Pretty(r.Id.Entry))} ({Safe(() => r.DynamicDescription.GetFormattedText(), "")})";
            case CardReward cr: return "Claim card reward (choose one of: " + string.Join(" | ", cr.Cards.Select(c => CardText(c))) + ")";
            case Reward rw: return "Claim reward: " + Safe(() => rw.Description.GetFormattedText(), rw.GetType().Name);
        }
        return a.Type switch
        {
            ActionType.EndTurn => "End turn",
            ActionType.Proceed => a.Label,
            ActionType.Skip => a.Label.Replace("alt ", "Card reward: ").Replace("skip relic", "Skip the relic"),
            ActionType.Confirm => "Confirm the selected cards",
            _ => a.Label,
        };
    }
}

using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Runs.History;
using static Sts2Ai.Shared.Encoding.ObsBuilder;

namespace Sts2Ai.Shared.Encoding;

/// <summary>
/// "Global memory" tokens built from the game's own per-floor run history (RunState.MapPointHistory),
/// which exists identically in the simulator and the live game:
///   * one summary token per act played so far (rooms by type, damage taken/healed, gold flow,
///     cards gained/removed/upgraded, card rewards skipped, potions used)
///   * one token per recent floor (encounter/event id, room type, damage taken, turns, hp after)
/// This gives the policy explicit long-horizon context (how hard the run has been, what the deck
/// went through, which elites were fought) without needing recurrence.
/// </summary>
public sealed class RunHistorySummary
{
    public int RecentFloors { get; init; } = 6;

    public void Encode(StateEncoder enc, IRunState run, Player player)
    {
        var b = enc.B;
        var hist = run.MapPointHistory;
        for (int act = 0; act < hist.Count; act++)
        {
            if (b.TokenCapacityLeft <= 8) return;
            int t = b.Token(TokenType.History, 0, 0, 1, act);
            var n = b.Num(t);
            foreach (var mp in hist[act])
            {
                foreach (var room in mp.Rooms)
                {
                    int k = room.RoomType switch
                    {
                        RoomType.Monster => 0, RoomType.Elite => 1, RoomType.Boss => 2, RoomType.Event => 3,
                        RoomType.Shop => 4, RoomType.RestSite => 5, RoomType.Treasure => 6, _ => 7,
                    };
                    n[k] += 1 / 5f;
                }
                var ps = mp.GetEntry(player.NetId);
                if (ps == null) continue;
                n[8] += ps.DamageTaken / 100f;
                n[9] += ps.HpHealed / 100f;
                n[10] += ps.GoldGained / 300f;
                n[11] += ps.GoldSpent / 300f;
                n[12] += ps.CardsGained.Count / 5f;
                n[13] += ps.CardsRemoved.Count / 3f;
                n[14] += ps.UpgradedCards.Count / 5f;
                n[15] += ps.CardChoices.Count(c => c.wasPicked) / 5f;
                n[16] += (ps.CardChoices.Count > 0 && !ps.CardChoices.Any(c => c.wasPicked) ? 1 : 0) / 5f;
                n[17] += ps.PotionUsed.Count / 5f;
                n[18] += ps.RelicChoices.Count / 5f;
                n[19] += ps.MaxHpGained / 20f;
                n[20] += ps.MaxHpLost / 20f;
                n[21] += ps.CardsTransformed.Count / 3f;
                n[22] += ps.CardsEnchanted.Count / 3f;
                n[23] += ps.GoldLost / 100f;
                n[24] = ps.CurrentHp / 100f;
                n[25] = ps.MaxHp / 100f;
            }
            n[26] = hist[act].Count / 17f;
        }

        // most recent floors, newest first
        int added = 0;
        for (int act = hist.Count - 1; act >= 0 && added < RecentFloors; act--)
        {
            for (int i = hist[act].Count - 1; i >= 0 && added < RecentFloors; i--)
            {
                if (b.TokenCapacityLeft <= 8) return;
                var mp = hist[act][i];
                var room = mp.Rooms.Count > 0 ? mp.Rooms[^1] : null;
                int t = b.Token(TokenType.History, enc.Vocab.Id(room?.ModelId), 0, 2 + (int)(room?.RoomType ?? RoomType.Unassigned), Math.Min(added, 127));
                var n = b.Num(t);
                n[0] = added / 10f;
                n[1] = room?.TurnsTaken / 10f ?? 0;
                var ps = mp.GetEntry(player.NetId);
                if (ps != null)
                {
                    n[2] = ps.DamageTaken / 50f;
                    n[3] = ps.HpHealed / 50f;
                    n[4] = ps.CurrentHp / 100f;
                    n[5] = ps.MaxHp > 0 ? ps.CurrentHp / (float)ps.MaxHp : 0;
                    n[6] = (ps.GoldGained - ps.GoldSpent - ps.GoldLost) / 200f;
                    n[7] = ps.CardsGained.Count / 3f;
                    n[8] = ps.PotionUsed.Count / 3f;
                }
                n[9] = (int)mp.MapPointType / 10f;
                added++;
            }
        }
    }
}

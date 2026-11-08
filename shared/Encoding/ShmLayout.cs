namespace Sts2Ai.Shared.Encoding;

/// <summary>
/// Byte layout of one environment's shared-memory observation slot (mirrors train/sts2ai/shm.py).
///   header  int32[32]
///   tokcat  int16[MaxTokens * NTokCat]
///   toknum  float32[MaxTokens * FTok]
///   actcat  int16[MaxActions * NActCat]
///   actref  int16[MaxActions * 2]
///   actnum  float32[MaxActions * FAct]
/// </summary>
public static class ShmLayout
{
    public const int HeaderInts = 32;
    public const int OffHeader = 0;
    public const int OffTokCat = OffHeader + HeaderInts * 4;
    public const int OffTokNum = OffTokCat + ObsSpec.MaxTokens * ObsSpec.NTokCat * 2;
    public const int OffActCat = OffTokNum + ObsSpec.MaxTokens * ObsSpec.FTok * 4;
    public const int OffActRef = OffActCat + ObsSpec.MaxActions * ObsSpec.NActCat * 2;
    public const int OffActNum = OffActRef + ObsSpec.MaxActions * 2 * 2;
    public const int SlotBytes = OffActNum + ObsSpec.MaxActions * ObsSpec.FAct * 4;

    // header indices
    public const int HSeq = 0, HDecision = 1, HNumTok = 2, HNumAct = 3, HDone = 4, HWon = 5, HAct = 6,
        HActFloor = 7, HTotalFloor = 8, HHp = 9, HMaxHp = 10, HGold = 11, HInCombat = 12, HCombatId = 13,
        HDeckSize = 14, HError = 15, HCharacter = 16, HAscension = 17, HBossKills = 18, HEliteKills = 19,
        HCombatHpStart = 20, HDroppedTok = 21, HDroppedAct = 22, HStepInRun = 23, HCombatsWon = 24,
        HRoomType = 25, HRoomModel = 26, // current room (at episode end: where the run ended)
        HRestart = 27, // 0: fresh run; >0: practice episode restarted from a recorded point (Worker situation + 1; 13 = a3_boss2)
        HCoop = 28, // co-op team size (0: single-player); HP fields are then team totals
        HPrefixActions = 29, // cumulative replay-prefix actions executed by this worker
        HRequestedRestart = 30, // requested situation + 1; 0 for fresh/unspecified legacy
        HAvailableRestarts = 31; // mask of nonempty restart pools: 12 bits, 13 with --boss2-situation (bit 12: a3_boss2)
}

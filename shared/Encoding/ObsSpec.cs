namespace Sts2Ai.Shared.Encoding;

/// <summary>Mirror of train/sts2ai/spec.py. Keep in sync; bump Version on any change.</summary>
public static class ObsSpec
{
    public const int Version = 1;
    public const int MaxTokens = 192;
    public const int MaxActions = 128;
    public const int FTok = 32;
    public const int FAct = 8;
    public const int NTokCat = 5;
    public const int NActCat = 3;
    public const int HashBuckets = 1024;
    public const int AuxVocab = 128;
}

public enum TokenType
{
    Pad = 0, Global = 1, HandCard = 2, DrawCard = 3, DiscardCard = 4, ExhaustCard = 5, DeckCard = 6,
    OfferCard = 7, Player = 8, Enemy = 9, Power = 10, Relic = 11, Potion = 12, Orb = 13, MapNode = 14,
    Option = 15, OfferRelic = 16, OfferPotion = 17, History = 18, Boss = 19, PlayPile = 20,
}

public enum ActionType
{
    Pad = 0, EndTurn = 1, PlayCard = 2, UsePotion = 3, DiscardPotion = 4, ChooseOption = 5, ChooseMapNode = 6,
    SelectCard = 7, Skip = 8, ClaimReward = 9, Buy = 10, RemoveCardService = 11, Proceed = 12, SelectRelic = 13,
    Confirm = 14,
}

public enum DecisionType
{
    None = 0, Combat = 1, CardSelect = 2, CardReward = 3, Rewards = 4, Map = 5, Event = 6, Rest = 7, Shop = 8,
    Treasure = 9, RelicSelect = 10, BundleSelect = 11, Other = 12,
}

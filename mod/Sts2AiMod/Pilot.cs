using System.Reflection;
using Godot;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Events.Custom;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.RestSite;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using Sts2Ai.Shared.Decisions;
using Sts2Ai.Shared.Encoding;

namespace Sts2AiMod;

/// <summary>
/// Finds the decision the live game is waiting for and knows how to carry out each choice through
/// the game's own UI nodes (the same interactions the game's AutoSlay test bot uses). Decisions are
/// built with the shared DecisionBuilder, so they are encoded exactly as in training.
/// </summary>
public sealed class Pilot(DecisionBuilder build)
{
    /// <summary>A pending decision plus how to execute one of its actions.</summary>
    public sealed record Pending(Decision Decision, string Context, Func<LegalAction, Task> Execute);

    static readonly FieldInfo CardRewardOptions = typeof(NCardRewardSelectionScreen).GetField("_options", BindingFlags.NonPublic | BindingFlags.Instance)!;
    static readonly FieldInfo CardRewardExtras = typeof(NCardRewardSelectionScreen).GetField("_extraOptions", BindingFlags.NonPublic | BindingFlags.Instance)!;
    static readonly FieldInfo BundleList = typeof(NChooseABundleSelectionScreen).GetField("_bundles", BindingFlags.NonPublic | BindingFlags.Instance)!;
    static readonly FieldInfo CardRewardAltContainer = typeof(NCardRewardSelectionScreen).GetField("_rewardAlternativesContainer", BindingFlags.NonPublic | BindingFlags.Instance)!;

    /// <summary>Null when nothing is awaiting a choice, or the screen is one the AI leaves to the player.</summary>
    public Pending? Detect(RunState state, Player player, out string waitingFor)
    {
        waitingFor = "";
        if (NGame.Instance?.Transition is { InTransition: true })
        {
            waitingFor = "transition";
            return null;
        }
        var overlay = NOverlayStack.Instance?.ScreenCount > 0 ? NOverlayStack.Instance.Peek() : null;
        // finished screens (rewards, Crystal Sphere, ...) stay in the overlay stack when proceeding opens the map
        // on top of them (the back arrow returns to them): while the map is open and accepts travel, the map is
        // the decision (as in AutoSlay)
        if (overlay != null && NMapScreen.Instance is { IsOpen: true, IsTravelEnabled: true } mapOverOverlay && mapOverOverlay.IsVisibleInTree())
            return MapScreen(state, player, mapOverOverlay, out waitingFor);
        switch (overlay)
        {
            case NGameOverScreen:
                waitingFor = "run over";
                return null;
            case NCardRewardSelectionScreen crs:
                return CardRewardScreen(state, player, crs);
            case NCardGridSelectionScreen grid:
                return Settled(grid) ? GridSelectScreen(state, player, grid) : null;
            case NChooseACardSelectionScreen choose:
                return Settled(choose) ? ChooseACardScreen(state, player, choose) : null;
            case NRewardsScreen rs:
                return RewardsScreen(state, player, rs);
            case NChooseABundleSelectionScreen bs:
                return BundleScreen(state, player, bs);
            case NCrystalSphereScreen cs:
                return CrystalSphere(cs);
            case Node n:
                waitingFor = "your choice: " + n.Name;
                return null;
        }
        if (NMapScreen.Instance is { IsOpen: true } map)
            return MapScreen(state, player, map, out waitingFor);
        if (CombatManager.Instance.IsInProgress)
        {
            if (NCombatRoom.Instance?.Ui?.Hand is { IsInCardSelection: true } hand)
                return Settled(hand) ? HandSelect(state, player, hand) : null;
            return AwaitingCombatInput(player) ? Combat(state, player) : null;
        }
        var room = state.CurrentRoom;
        var roomNode = ((SceneTree)Engine.GetMainLoop()).Root.GetNodeOrNull("/root/Game/RootSceneContainer/Run/RoomContainer");
        if (roomNode == null) return null;
        return room switch
        {
            EventRoom er => Event(state, player, er, roomNode),
            RestSiteRoom rs => Rest(state, player, rs, roomNode),
            MerchantRoom mr => Shop(state, player, mr, roomNode),
            TreasureRoom => Treasure(state, player, roomNode),
            _ => null,
        };
    }

    // ------------------------------------------------------------------ combat

    public static bool AwaitingCombatInput(Player player)
    {
        var cm = CombatManager.Instance;
        var rm = RunManager.Instance;
        var pcs = player.PlayerCombatState;
        if (!(cm.IsInProgress && !cm.IsOverOrEnding && pcs != null && pcs.Phase == PlayerTurnPhase.Play
              && !cm.PlayerActionsDisabled && !cm.IsPlayerReadyToEndTurn(player)))
            return false;
        if (rm.NetService is { Type: MegaCrit.Sts2.Core.Multiplayer.Game.NetGameType.Singleplayer })
            return rm.ActionQueueSet.IsEmpty && !rm.ActionExecutor.IsRunning;
        // multiplayer: teammates' card plays keep the shared queues busy for most of the turn; like the game's own
        // hand UI, only wait for our own actions
        if (cm.PlayersTakingExtraTurn.Count > 0 && !cm.PlayersTakingExtraTurn.Contains(player)) return false;
        if (rm.ActionExecutor.CurrentlyRunningAction is { } cur && cur.OwnerId == player.NetId) return false;
        return OwnQueueEmpty(rm.ActionQueueSet, player.NetId);
    }

    static readonly FieldInfo QueuesField = typeof(ActionQueueSet).GetField("_actionQueues", BindingFlags.NonPublic | BindingFlags.Instance)!;
    static FieldInfo? _queueOwner, _queueActions;

    static bool OwnQueueEmpty(ActionQueueSet set, ulong owner)
    {
        if (QueuesField.GetValue(set) is not System.Collections.IEnumerable queues) return set.IsEmpty;
        foreach (var q in queues)
        {
            _queueOwner ??= q.GetType().GetField("ownerId");
            _queueActions ??= q.GetType().GetField("actions");
            if (_queueOwner?.GetValue(q) is ulong id && id == owner)
                return _queueActions?.GetValue(q) is not System.Collections.ICollection c || c.Count == 0;
        }
        return true;
    }

    static bool Multiplayer => RunManager.Instance.NetService is not { Type: MegaCrit.Sts2.Core.Multiplayer.Game.NetGameType.Singleplayer };

    Pending Combat(RunState state, Player player) => new(build.Combat(state, player), "combat", a =>
    {
        switch (a.Type)
        {
            case ActionType.PlayCard: ((CardModel)a.Payload!).TryManualPlay((Creature?)a.Target); break;
            case ActionType.UsePotion: ((PotionModel)a.Payload!).EnqueueManualUse((Creature?)a.Target); break;
            case ActionType.DiscardPotion:
                RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new DiscardPotionGameAction(player, (uint)a.Index, true));
                break;
            case ActionType.EndTurn:
                RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new EndPlayerTurnAction(player, player.PlayerCombatState!.TurnNumber));
                break;
        }
        return Task.CompletedTask;
    });

    // ------------------------------------------------------------------ overlays

    Pending? CardRewardScreen(RunState state, Player player, NCardRewardSelectionScreen screen)
    {
        if (CardRewardOptions.GetValue(screen) is not IReadOnlyList<CardCreationResult> options) return null;
        var extras = CardRewardExtras.GetValue(screen) as IReadOnlyList<CardRewardAlternative> ?? Array.Empty<CardRewardAlternative>();
        var d = build.CardReward(state, player, options.Select(o => o.Card).ToList(), extras);
        return new(d, "card reward", a =>
        {
            if (a.Type == ActionType.SelectCard)
            {
                var holder = screen.GetCardHolder((CardModel)a.Payload!);
                holder.EmitSignal(NCardHolder.SignalName.Pressed, holder);
            }
            else
            {
                build.NoteCardRewardChoice(a);
                // alternatives (skip, ...) are buttons in the same order as the extra options
                var container = (Node)CardRewardAltContainer.GetValue(screen)!;
                var buttons = container.GetChildren().OfType<NClickableControl>().ToList();
                int i = a.Payload is CardRewardAlternative alt ? extras.ToList().IndexOf(alt) : 0;
                if (i >= 0 && i < buttons.Count) buttons[i].ForceClick();
            }
            return Task.CompletedTask;
        });
    }

    Pending? BundleScreen(RunState state, Player player, NChooseABundleSelectionScreen screen)
    {
        if (BundleList.GetValue(screen) is not IReadOnlyList<IReadOnlyList<CardModel>> bundles) return null;
        var nodes = FindAll<NCardBundle>(screen);
        var d = build.Bundle(state, player, bundles);
        return new(d, "card bundle", async a =>
        {
            if (a.Index >= 0 && a.Index < nodes.Count) nodes[a.Index].Hitbox.ForceClick();
            await Task.Delay(500);
            FindFirst<NConfirmButton>(screen)?.ForceClick();
        });
    }

    /// <summary>Crystal Sphere minigame: not a policy decision (the simulator uses a heuristic too); reveal
    /// hidden cells until the proceed button appears, like the game's own AutoSlay.</summary>
    Pending? CrystalSphere(NCrystalSphereScreen screen)
    {
        var proceed = screen.GetNodeOrNull<NProceedButton>("%ProceedButton");
        if (proceed is { IsEnabled: true })
            return new(new Decision { Type = DecisionType.Event }, "crystal sphere: done", _ => { proceed.ForceClick(); return Task.CompletedTask; });
        var cellsNode = screen.GetNodeOrNull<Control>("%Cells");
        if (cellsNode == null) return null;
        var hidden = FindAll<NCrystalSphereCell>(cellsNode).Where(c => c.Visible && c.Entity.IsHidden).ToList();
        if (hidden.Count == 0) return null;
        var cell = hidden[hidden.Count / 2];
        return new(new Decision { Type = DecisionType.Event }, "crystal sphere", _ =>
        {
            cell.EmitSignal(NClickableControl.SignalName.Released, cell);
            return Task.CompletedTask;
        });
    }

    // rewards already opened on the current rewards screen (as in the simulator's OfferRewards): a card
    // reward that was skipped stays on the screen, and offering it again would loop claim -> skip forever
    readonly HashSet<Reward> _triedRewards = new(ReferenceEqualityComparer.Instance);
    NRewardsScreen? _triedScreen;

    Pending? RewardsScreen(RunState state, Player player, NRewardsScreen screen)
    {
        if (!ReferenceEquals(screen, _triedScreen)) { _triedScreen = screen; _triedRewards.Clear(); }
        var buttons = FindAll<NRewardButton>(screen).Where(b => b.IsEnabled && b.Reward != null && !_triedRewards.Contains(b.Reward)).ToList();
        // gold is claimed without asking the policy, exactly as in training
        if (buttons.FirstOrDefault(b => DecisionBuilder.AutoClaim(b.Reward!)) is { } gold)
            return new(new Decision { Type = DecisionType.Rewards }, "claim gold", _ => { gold.ForceClick(); return Task.CompletedTask; });
        var open = buttons.Select(b => b.Reward!).Where(r => !(r is PotionReward && !player.HasOpenPotionSlots)).ToList();
        var waiting = buttons.Select(b => b.Reward).OfType<PotionReward>().Where(r => r.Potion != null).ToList();
        bool potionWaiting = waiting.Count > 0 && !player.HasOpenPotionSlots;
        var proceed = FindFirst<NProceedButton>(screen);
        if (proceed is { IsEnabled: false }) proceed = null; // still animating in: clicks would be ignored
        var d = build.Rewards(state, player, open, potionWaiting ? waiting : null, allowProceed: proceed != null);
        return new(d, "rewards", async a =>
        {
            switch (a.Type)
            {
                case ActionType.ClaimReward:
                    _triedRewards.Add((Reward)a.Payload!);
                    buttons.FirstOrDefault(b => ReferenceEquals(b.Reward, a.Payload))?.ForceClick();
                    break;
                case ActionType.DiscardPotion:
                    await PotionCmd.Discard((PotionModel)a.Payload!);
                    break;
                case ActionType.Proceed:
                    proceed?.ForceClick();
                    break;
            }
        });
    }

    // ------------------------------------------------------------------ card selection through the UI (multiplayer)
    // In single-player the ICardSelector hook answers these prompts (as in training). In multiplayer that hook would
    // skip the choice synchronization, so the game's own screens appear and are answered here with the same
    // decisions (DecisionBuilder.CardSelect): pick cards one at a time, then confirm.

    static readonly BindingFlags Priv = BindingFlags.NonPublic | BindingFlags.Instance;
    static readonly FieldInfo GridCards = typeof(NCardGridSelectionScreen).GetField("_cards", Priv)!;
    static readonly MethodInfo GridClick = typeof(NCardGridSelectionScreen).GetMethod("OnCardClicked", Priv)!;
    static readonly FieldInfo ChooseCards = typeof(NChooseACardSelectionScreen).GetField("_cards", Priv)!;
    static readonly FieldInfo ChooseCanSkip = typeof(NChooseACardSelectionScreen).GetField("_canSkip", Priv)!;
    static readonly FieldInfo ChooseSkip = typeof(NChooseACardSelectionScreen).GetField("_skipButton", Priv)!;
    static readonly FieldInfo HandPrefs = typeof(NPlayerHand).GetField("_prefs", Priv)!;
    static readonly FieldInfo HandSelected = typeof(NPlayerHand).GetField("_selectedCards", Priv)!;
    static readonly FieldInfo HandFilter = typeof(NPlayerHand).GetField("_currentSelectionFilter", Priv)!;
    static readonly FieldInfo HandConfirm = typeof(NPlayerHand).GetField("_selectModeConfirmButton", Priv)!;
    static readonly MethodInfo HandPress = typeof(NPlayerHand).GetMethod("OnHolderPressed", Priv)!;

    // selection screens ignore clicks while opening (choose-a-card: 350 ms): act once a screen has been up a moment
    GodotObject? _settling;
    DateTime _settlingSince;
    bool Settled(GodotObject screen)
    {
        if (!ReferenceEquals(screen, _settling)) { _settling = screen; _settlingSince = DateTime.UtcNow; }
        return (DateTime.UtcNow - _settlingSince).TotalSeconds >= 0.6;
    }

    Pending? GridSelectScreen(RunState state, Player player, NCardGridSelectionScreen screen)
    {
        if (GridCards.GetValue(screen) is not IReadOnlyList<CardModel> cards || cards.Count == 0) return null;
        var t = screen.GetType();
        var picked = (t.GetField("_selectedCards", Priv)?.GetValue(screen) as IEnumerable<CardModel>)?.ToList() ?? new List<CardModel>();
        var prefs = t.GetField("_prefs", Priv)?.GetValue(screen) is CardSelectorPrefs p ? p : default;
        int max = Math.Min(Math.Max(prefs.MaxSelect, 1), cards.Count), min = Math.Min(prefs.MinSelect, max);
        NConfirmButton? Confirm()
        {
            var preview = screen.GetNodeOrNull<Control>("%PreviewContainer");
            var previewConfirm = preview?.GetNodeOrNull<NConfirmButton>("%PreviewConfirm") ?? screen.GetNodeOrNull<NConfirmButton>("%PreviewConfirm");
            if (preview is { Visible: true } && previewConfirm is { IsEnabled: true }) return previewConfirm;
            var confirm = screen.GetNodeOrNull<NConfirmButton>("%Confirm");
            return confirm is { IsEnabled: true } ? confirm : null;
        }
        if (picked.Count >= max)
        {
            // enough cards picked: confirm (deck screens show a preview with its own confirm button)
            var c = Confirm();
            if (c == null) { StuckLog("card select confirm", () => $"{t.Name} picked={picked.Count}/{max}: no enabled confirm"); return null; }
            return new(new Decision { Type = DecisionType.CardSelect }, "card select: confirm", _ => { c.ForceClick(); return Task.CompletedTask; });
        }
        var d = build.CardSelect(state, player, cards, picked, min, max);
        return new(d, "card select (" + t.Name + ")", a =>
        {
            if (a.Type == ActionType.Confirm) Confirm()?.ForceClick();
            else GridClick.Invoke(screen, new object[] { (CardModel)a.Payload! });
            return Task.CompletedTask;
        });
    }

    Pending? ChooseACardScreen(RunState state, Player player, NChooseACardSelectionScreen screen)
    {
        if (ChooseCards.GetValue(screen) is not IReadOnlyList<CardModel> cards || cards.Count == 0) return null;
        bool canSkip = ChooseCanSkip.GetValue(screen) is true;
        // training (selector path) always allowed skipping here (min 0); the screen only does when canSkip
        var d = build.CardSelect(state, player, cards, Array.Empty<CardModel>(), canSkip ? 0 : 1, 1);
        return new(d, "choose a card", a =>
        {
            if (a.Type == ActionType.Confirm)
                (ChooseSkip.GetValue(screen) as NClickableControl)?.ForceClick();
            else if (FindAll<NCardHolder>(screen).FirstOrDefault(h => ReferenceEquals(h.CardModel, a.Payload)) is { } holder)
                holder.EmitSignal(NCardHolder.SignalName.Pressed, holder);
            return Task.CompletedTask;
        });
    }

    Pending? HandSelect(RunState state, Player player, NPlayerHand hand)
    {
        var prefs = HandPrefs.GetValue(hand) is CardSelectorPrefs p ? p : default;
        var picked = (HandSelected.GetValue(hand) as IEnumerable<CardModel>)?.ToList() ?? new List<CardModel>();
        var filter = HandFilter.GetValue(hand) as Func<CardModel, bool>;
        var pcs = player.PlayerCombatState;
        if (pcs == null) return null;
        var options = pcs.Hand.Cards.Where(c => filter == null || filter(c)).ToList();
        if (options.Count == 0) return null;
        int max = Math.Min(Math.Max(prefs.MaxSelect, 1), options.Count), min = Math.Min(prefs.MinSelect, max);
        var confirm = HandConfirm.GetValue(hand) as NConfirmButton;
        if (picked.Count >= max)
        {
            if (confirm is not { IsEnabled: true }) { StuckLog("hand select confirm", () => $"picked={picked.Count}/{max}: confirm disabled"); return null; }
            return new(new Decision { Type = DecisionType.CardSelect }, "hand select: confirm", _ => { confirm.ForceClick(); return Task.CompletedTask; });
        }
        var holders = FindAll<NHandCardHolder>(hand);
        var d = build.CardSelect(state, player, options, picked, min, max);
        return new(d, "hand select", a =>
        {
            if (a.Type == ActionType.Confirm) { if (confirm is { IsEnabled: true }) confirm.ForceClick(); }
            else if (holders.FirstOrDefault(h => ReferenceEquals(h.CardModel, a.Payload)) is { } holder)
                HandPress.Invoke(hand, new object[] { holder });
            return Task.CompletedTask;
        });
    }

    // ------------------------------------------------------------------ map

    /// <summary>
    /// The next map node, once the map accepts travel (open, visible, travel enabled, not travelling, the
    /// nodes enabled: clicks before that are ignored). The simulator's "next act" step has no map action in
    /// the game: the boss rewards' proceed button starts the act change, so that decision is only waited out.
    /// </summary>
    Pending? MapScreen(RunState state, Player player, NMapScreen map, out string waitingFor)
    {
        waitingFor = "map";
        if (!map.IsVisibleInTree() || !map.IsTravelEnabled || map.IsTraveling)
        {
            waitingFor = "map not ready";
            StuckLog("map", () => $"visible={map.IsVisibleInTree()} travelEnabled={map.IsTravelEnabled} traveling={map.IsTraveling}");
            return null;
        }
        // multiplayer: a map click is a vote; clicking our own vote again pings the other players
        if (Multiplayer && RunManager.Instance.MapSelectionSynchronizer.GetVote(player) != null) { waitingFor = "map vote cast"; return null; }
        var d = build.Map(state, player);
        if (d.Actions.Any(a => a.Type == ActionType.Proceed)) { waitingFor = "act transition"; return null; }
        var points = FindAll<NMapPoint>(map);
        NMapPoint? NodeFor(LegalAction a) => points.FirstOrDefault(p => ReferenceEquals(p.Point, a.Payload) || p.Point.coord.Equals(((MapPoint)a.Payload!).coord));
        if (d.Actions.Where(a => a.Type == ActionType.ChooseMapNode).Any(a => NodeFor(a) is not { IsEnabled: true }))
        {
            waitingFor = "map not ready";
            StuckLog("map nodes", () => string.Join("; ", d.Actions.Where(a => a.Type == ActionType.ChooseMapNode)
                .Select(a => $"{a.Label} node={(NodeFor(a) is { } n ? $"enabled={n.IsEnabled} visible={n.Visible}" : "missing")}")));
            return null;
        }
        return new(d, "map", a =>
        {
            if (a.Type == ActionType.ChooseMapNode)
                NodeFor(a)?.ForceClick();
            else if (a.Type == ActionType.UsePotion)
                ((PotionModel)a.Payload!).EnqueueManualUse((Creature?)a.Target);
            return Task.CompletedTask;
        });
    }

    // ------------------------------------------------------------------ rooms

    Pending? Event(RunState state, Player player, EventRoom er, Node roomNode)
    {
        var ev = er.LocalMutableEvent;
        if (FindFirst<NFakeMerchant>(roomNode) is { } fake) return FakeMerchant(fake);
        // Ancient layouts (Neow, ...) make options clickable with EnableButton() (mouse filter Stop) and never
        // touch IsEnabled; regular layouts use Enable()/Disable()
        var ancient = FindFirst<NAncientEventLayout>(roomNode);
        var hitbox = ancient?.GetNodeOrNull<NButton>("%DialogueHitbox");
        bool dialogue = hitbox is { Visible: true, IsEnabled: true };
        var all = FindAll<NEventOptionButton>(roomNode);
        var buttons = all
            .Where(b => !b.Option.IsLocked && b.Visible
                        && (ancient != null ? !dialogue && b.MouseFilter == Control.MouseFilterEnum.Stop : b.IsEnabled))
            .ToList();
        // a finished event has no options of its own (SetEventFinished): the room shows its own proceed button
        var proceedButton = all.FirstOrDefault(b => b.Option.IsProceed && !b.Option.IsLocked && b.Visible && !dialogue
                                                    && (b.IsEnabled || b.MouseFilter == Control.MouseFilterEnum.Stop));
        if (buttons.Count == 0)
        {
            // Ancient events: click through the dialogue until the options appear
            if (dialogue && hitbox != null)
                return new(new Decision { Type = DecisionType.Event }, "event dialogue", _ =>
                {
                    hitbox.EmitSignal(NClickableControl.SignalName.Released, hitbox);
                    return Task.CompletedTask;
                });
            return null;
        }
        // shared events: options are votes; once ours is in, wait for the others
        var events = RunManager.Instance.EventSynchronizer;
        if (Multiplayer && events.IsShared && events.GetPlayerVote(player) != null) return null;
        var d = build.Event(state, player, ev);
        if (d.Actions.Count == 0 || d.Actions.All(a => ((MegaCrit.Sts2.Core.Events.EventOption)a.Payload!).IsProceed))
        {
            // only "proceed" left: taken without asking the policy, as in the simulator
            if (proceedButton == null) { LogEventState(ev, all, dialogue); return null; }
            return new(new Decision { Type = DecisionType.Event }, "event proceed", _ =>
            {
                proceedButton.ForceClick();
                return Task.CompletedTask;
            });
        }
        return new(d, "event " + ev.Id.Entry, a =>
        {
            buttons.FirstOrDefault(b => ReferenceEquals(b.Option, a.Payload))?.ForceClick();
            return Task.CompletedTask;
        });
    }

    /// <summary>FakeMerchant, the only custom-layout event: no option buttons, just a relic counter and its own
    /// proceed button (which opens the map). It has no options, so the simulator treats it as finished and leaves
    /// at once: the policy never learned to shop here. Leave, as AutoSlay does; close the counter first if the
    /// player opened it.</summary>
    Pending? FakeMerchant(NFakeMerchant fake)
    {
        if (fake.Inventory is { IsOpen: true } inv)
        {
            var back = FindFirst<NBackButton>(inv);
            if (back is not { IsEnabled: true }) return null;
            return new(new Decision { Type = DecisionType.Event }, "fake merchant: close counter", _ =>
            {
                back.ForceClick();
                return Task.CompletedTask;
            });
        }
        var proceed = fake.GetNodeOrNull<NProceedButton>("%ProceedButton");
        if (proceed is not { IsEnabled: true, Visible: true })
        {
            StuckLog("fake merchant", () => $"proceed={(proceed == null ? "missing" : $"enabled={proceed.IsEnabled} visible={proceed.Visible}")}");
            return null;
        }
        return new(new Decision { Type = DecisionType.Event }, "fake merchant: leave", _ =>
        {
            proceed.ForceClick();
            return Task.CompletedTask;
        });
    }

    Pending? Rest(RunState state, Player player, RestSiteRoom rs, Node roomNode)
    {
        var node = FindFirst<NRestSiteRoom>(roomNode);
        if (node == null) return null;
        var buttons = FindAll<NRestSiteButton>(node);
        // nothing clickable yet (room fading in, or an option's animation still running): wait
        if (!node.ProceedButton.IsEnabled && !buttons.Any(b => b.IsEnabled && b.Visible)) return null;
        // Mend (heal an ally) needs UI targeting: unavailable, as in training (where co-op disables it)
        var d = build.Rest(state, player, rs.Options, o => o is not MendRestSiteOption);
        return new(d, "rest site", a =>
        {
            if (a.Type == ActionType.ChooseOption)
                buttons.FirstOrDefault(b => ReferenceEquals(b.Option, a.Payload))?.ForceClick();
            else if (node.ProceedButton.IsEnabled)
                node.ProceedButton.ForceClick();
            return Task.CompletedTask;
        });
    }

    DateTime _shopOpenedAt = DateTime.MinValue;

    Pending? Shop(RunState state, Player player, MerchantRoom mr, Node roomNode)
    {
        var node = FindFirst<NMerchantRoom>(roomNode);
        if (node == null) return null;
        // open the inventory as its own step and let it finish opening before buying (AutoSlay waits 500 ms):
        // purchases animate cards and items out of the inventory's slots
        if (node.Inventory is { IsOpen: false })
            return new(new Decision { Type = DecisionType.Shop }, "open shop", _ =>
            {
                node.OpenInventory();
                _shopOpenedAt = DateTime.UtcNow;
                return Task.CompletedTask;
            });
        if ((DateTime.UtcNow - _shopOpenedAt).TotalSeconds < 0.8) return null;
        var inv = mr.GetLocalInventory();
        var d = build.Shop(state, player, inv);
        return new(d, "shop", async a =>
        {
            switch (a.Type)
            {
                case ActionType.Buy:
                case ActionType.RemoveCardService:
                    // not awaited: a purchase completes only after the item's pickup effect, and some open a
                    // screen the AI itself must answer (Cauldron: 5 potions), which would deadlock
                    Observe(((MerchantEntry)a.Payload!).OnTryPurchaseWrapper(inv), "purchase");
                    break;
                case ActionType.Proceed:
                    FindFirst<NBackButton>(node)?.ForceClick();
                    await Task.Delay(300);
                    node.ProceedButton.ForceClick();
                    break;
            }
        });
    }

    Pending? Treasure(RunState state, Player player, Node roomNode)
    {
        var node = FindFirst<NTreasureRoom>(roomNode);
        if (node == null) return null;
        // the relic is rolled on entering the room, before the chest is opened: open the chest first
        var chest = node.GetNodeOrNull<NClickableControl>("%Chest") ?? node.GetNodeOrNull<NClickableControl>("Chest");
        if (chest == null) StuckLog("treasure chest", () => "chest node not found under " + node.GetPath());
        if (chest is { IsEnabled: true, Visible: true })
            return new(new Decision { Type = DecisionType.Treasure }, "open chest", _ => { chest.ForceClick(); return Task.CompletedTask; });
        var relics = RunManager.Instance.TreasureRoomRelicSynchronizer.CurrentRelics;
        if (relics == null || relics.Count == 0)
        {
            if (node.ProceedButton.IsEnabled)
                return new(new Decision { Type = DecisionType.Treasure }, "leave treasure room", _ => { node.ProceedButton.ForceClick(); return Task.CompletedTask; });
            return null;
        }
        // multiplayer: the relic pick is a vote; once ours is in, wait for the others
        if (Multiplayer && RunManager.Instance.TreasureRoomRelicSynchronizer.GetPlayerVote(player).voteReceived) return null;
        var holders = FindAll<NTreasureRoomRelicHolder>(node);
        // relics fly in after the chest opens; clicks before the holders are ready are ignored
        if (!holders.Any(h => h.IsEnabled && h.Visible)) return null;
        var d = build.Treasure(state, player, relics);
        return new(d, "treasure", a =>
        {
            if (a.Type == ActionType.SelectRelic)
                holders.FirstOrDefault(h => h.Index == a.Index && h.IsEnabled)?.ForceClick();
            else if (node.ProceedButton.IsEnabled)
                node.ProceedButton.ForceClick();
            else
                holders.FirstOrDefault(h => h.IsEnabled)?.ForceClick(); // proceeding requires a pick
            return Task.CompletedTask;
        });
    }

    static void Observe(Task t, string what) =>
        t.ContinueWith(x => Log.Error($"[sts2ai] {what} failed: {x.Exception}"), TaskContinuationOptions.OnlyOnFaulted);

    readonly Dictionary<string, DateTime> _stuckSince = new();

    /// <summary>The same waiting state for more than 5 s: log details once (for live debugging).</summary>
    void StuckLog(string key, Func<string> detail)
    {
        var now = DateTime.UtcNow;
        foreach (var k in _stuckSince.Keys.Where(k => k != key && (now - _stuckSince[k]).TotalSeconds > 30).ToList()) _stuckSince.Remove(k);
        if (!_stuckSince.TryGetValue(key, out var since)) { _stuckSince[key] = now; return; }
        if (since == DateTime.MaxValue || (now - since).TotalSeconds < 5) return;
        _stuckSince[key] = DateTime.MaxValue;
        Log.Info($"[sts2ai] waiting on {key}: {detail()}");
    }

    DateTime _eventStuckSince = DateTime.MaxValue;
    string _eventStuckKey = "";

    /// <summary>Nothing to click in an event room for a while: log what is there (for live debugging).</summary>
    void LogEventState(EventModel ev, List<NEventOptionButton> buttons, bool dialogue)
    {
        string key = ev.Id.Entry + "|" + ev.IsFinished + "|" + buttons.Count;
        if (key != _eventStuckKey) { _eventStuckKey = key; _eventStuckSince = DateTime.UtcNow; return; }
        if ((DateTime.UtcNow - _eventStuckSince).TotalSeconds < 5) return;
        _eventStuckSince = DateTime.MaxValue;
        Log.Info($"[sts2ai] event {ev.Id.Entry} waiting: finished={ev.IsFinished} options={ev.CurrentOptions.Count} dialogue={dialogue} buttons=" +
                 string.Join("; ", buttons.Select(b => $"{b.Option.TextKey} proceed={b.Option.IsProceed} locked={b.Option.IsLocked} visible={b.Visible} enabled={b.IsEnabled} mouse={b.MouseFilter}")));
    }

    // ------------------------------------------------------------------ node helpers

    static List<T> FindAll<T>(Node start) where T : Node
    {
        var found = new List<T>();
        void Walk(Node n)
        {
            if (!GodotObject.IsInstanceValid(n)) return;
            if (n is T t) found.Add(t);
            foreach (var c in n.GetChildren()) Walk(c);
        }
        if (GodotObject.IsInstanceValid(start)) Walk(start);
        return found;
    }

    static T? FindFirst<T>(Node start) where T : Node
    {
        if (!GodotObject.IsInstanceValid(start)) return null;
        if (start is T t) return t;
        foreach (var c in start.GetChildren())
            if (FindFirst<T>(c) is { } r) return r;
        return null;
    }
}

using Godot;

namespace Sts2AiMod;

/// <summary>Small always-on-top panel: autoplay state, P(win), estimated final floor, top choices, and buttons
/// for autoplay / step / hide (the F8 / F9 / F10 keys do not reach the game in some remote-desktop sessions).
/// When hidden, a small "STS2 AI" button stays in the corner to bring the panel back.</summary>
public sealed class Overlay
{
    readonly PanelContainer _panel;
    readonly Label _label;
    readonly Button _auto, _show, _search;
    string _last = "";
    bool? _lastAuto;
    string _lastSearch = "";

    Overlay(PanelContainer panel, Label label, Button auto, Button show, Button search)
    {
        _panel = panel;
        _label = label;
        _auto = auto;
        _show = show;
        _search = search;
    }

    static Button MakeButton(string text, Action onClick)
    {
        // no keyboard focus: a focused button would be pressed again by Space/Enter meant for the game
        var b = new Button { Text = text, FocusMode = Control.FocusModeEnum.None, MouseFilter = Control.MouseFilterEnum.Stop };
        b.AddThemeFontSizeOverride("font_size", 15);
        b.Pressed += onClick;
        return b;
    }

    // below the game's top bar (map / deck / settings buttons sit in the top-right corner): the buttons here take
    // clicks, so they must not cover the game's own
    const float Top = 96;

    public static Overlay Create(Node root, AiController ai)
    {
        var layer = new CanvasLayer { Layer = 120, Name = "Sts2AiOverlay" };
        var panel = new PanelContainer
        {
            AnchorLeft = 1, AnchorRight = 1, AnchorTop = 0, AnchorBottom = 0,
            OffsetLeft = -430, OffsetRight = -12, OffsetTop = Top,
            MouseFilter = Control.MouseFilterEnum.Ignore,
        };
        var style = new StyleBoxFlat
        {
            BgColor = new Color(0.05f, 0.05f, 0.08f, 0.82f),
            CornerRadiusBottomLeft = 6, CornerRadiusBottomRight = 6, CornerRadiusTopLeft = 6, CornerRadiusTopRight = 6,
            ContentMarginLeft = 10, ContentMarginRight = 10, ContentMarginTop = 8, ContentMarginBottom = 8,
        };
        panel.AddThemeStyleboxOverride("panel", style);
        // only the buttons take clicks; everything else lets them through to the game
        var column = new VBoxContainer { MouseFilter = Control.MouseFilterEnum.Ignore };
        var buttons = new HBoxContainer { MouseFilter = Control.MouseFilterEnum.Ignore };
        var auto = MakeButton("Autoplay: OFF", ai.ToggleAutoplay);
        auto.SizeFlagsHorizontal = Control.SizeFlags.ExpandFill;
        buttons.AddChild(auto);
        buttons.AddChild(MakeButton("Step", ai.RequestStep));
        var search = MakeButton("Search: n/a", ai.ToggleSearch);
        buttons.AddChild(search);
        buttons.AddChild(MakeButton("Hide", ai.ToggleVisible));
        var label = new Label { MouseFilter = Control.MouseFilterEnum.Ignore, AutowrapMode = TextServer.AutowrapMode.WordSmart };
        label.AddThemeFontSizeOverride("font_size", 15);
        label.AddThemeColorOverride("font_color", new Color(0.92f, 0.92f, 0.95f));
        column.AddChild(buttons);
        column.AddChild(label);
        panel.AddChild(column);
        var show = MakeButton("STS2 AI", ai.ToggleVisible);
        show.AnchorLeft = show.AnchorRight = 1;
        show.OffsetLeft = -100;
        show.OffsetRight = -12;
        show.OffsetTop = Top;
        show.Visible = false;
        layer.AddChild(panel);
        layer.AddChild(show);
        root.CallDeferred(Node.MethodName.AddChild, layer);
        return new Overlay(panel, label, auto, show, search);
    }

    public void Update(AiController ai)
    {
        _panel.Visible = ai.Visible;
        _show.Visible = !ai.Visible;
        if (!ai.Visible) return;
        if (_lastAuto != ai.Autoplay)
        {
            _lastAuto = ai.Autoplay;
            _auto.Text = ai.Autoplay ? "Autoplay: ON" : "Autoplay: OFF";
            _auto.Modulate = ai.Autoplay ? new Color(0.55f, 1f, 0.55f) : Colors.White;
        }
        string st = !ai.SearchAvailable ? "Search: n/a" : ai.SearchEnabled ? "Search: ON" : "Search: OFF";
        if (st != _lastSearch)
        {
            _lastSearch = st;
            _search.Text = st;
            _search.Modulate = ai.SearchEnabled ? new Color(0.55f, 0.8f, 1f) : Colors.White;
        }
        var lines = new List<string>
        {
            $"STS2 AI · {ai.ModelName}   [F8 · F9 · F10]",
        };
        var o = ai.Output;
        if (o != null)
        {
            lines.Add($"P(win) {o.PWin * 100:F1}%    expected final floor {o.Floor * 50:F0}");
            lines.Add(ai.Context);
            var d = ai.Current;
            if (d != null && d.Actions.Count > 0 && o.Probs.Length == d.Actions.Count)
            {
                foreach (int i in Enumerable.Range(0, d.Actions.Count).OrderByDescending(i => o.Probs[i]).Take(4))
                    lines.Add($"  {o.Probs[i] * 100,5:F1}%  {d.Actions[i]}");
            }
        }
        else lines.Add(ai.Context);
        if (ai.SearchStatus.Length > 0) lines.Add(ai.SearchStatus);
        var text = string.Join("\n", lines);
        if (text != _last) { _label.Text = text; _last = text; }
    }
}

using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using Sts2Ai.Shared.Encoding;

namespace Sts2Sim.Engine;

/// <summary>Writes an encoded decision into this worker's shared-memory slot.</summary>
public sealed class ShmWriter : IDisposable
{
    readonly MemoryMappedFile _mmf;
    readonly MemoryMappedViewAccessor _view;
    readonly int[] _header = new int[ShmLayout.HeaderInts];
    readonly short[] _tc = new short[ObsSpec.MaxTokens * ObsSpec.NTokCat];
    readonly short[] _ac = new short[ObsSpec.MaxActions * ObsSpec.NActCat];
    readonly short[] _ar = new short[ObsSpec.MaxActions * 2];
    int _seq;

    public int[] Header => _header;

    public ShmWriter(string path)
    {
        _mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, ShmLayout.SlotBytes, MemoryMappedFileAccess.ReadWrite);
        _view = _mmf.CreateViewAccessor(0, ShmLayout.SlotBytes);
    }

    public void Write(ObsBuilder b)
    {
        int nt = b.NumTokens, na = b.NumActions;
        for (int i = 0; i < nt * ObsSpec.NTokCat; i++) _tc[i] = (short)b.TokCat[i];
        for (int i = 0; i < na * ObsSpec.NActCat; i++) _ac[i] = (short)b.ActCat[i];
        for (int i = 0; i < na * 2; i++) _ar[i] = (short)b.ActRef[i];
        _view.WriteArray(ShmLayout.OffTokCat, _tc, 0, nt * ObsSpec.NTokCat);
        _view.WriteArray(ShmLayout.OffTokNum, b.TokNum, 0, nt * ObsSpec.FTok);
        _view.WriteArray(ShmLayout.OffActCat, _ac, 0, na * ObsSpec.NActCat);
        _view.WriteArray(ShmLayout.OffActRef, _ar, 0, na * 2);
        _view.WriteArray(ShmLayout.OffActNum, b.ActNum, 0, na * ObsSpec.FAct);
        _header[ShmLayout.HNumTok] = nt;
        _header[ShmLayout.HNumAct] = na;
        _header[ShmLayout.HDecision] = (int)b.Decision;
        _header[ShmLayout.HDroppedTok] = b.DroppedTokens;
        _header[ShmLayout.HDroppedAct] = b.DroppedActions;
        _header[ShmLayout.HSeq] = ++_seq;
        _view.WriteArray(ShmLayout.OffHeader, _header, 0, _header.Length);
        _view.Flush();
    }

    /// <summary>Terminal / error frame with no tokens or actions.</summary>
    public void WriteHeaderOnly()
    {
        _header[ShmLayout.HNumTok] = 0;
        _header[ShmLayout.HNumAct] = 0;
        _header[ShmLayout.HSeq] = ++_seq;
        _view.WriteArray(ShmLayout.OffHeader, _header, 0, _header.Length);
        _view.Flush();
    }

    public void Dispose()
    {
        _view.Dispose();
        _mmf.Dispose();
    }
}

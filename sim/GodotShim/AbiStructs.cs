using System.Runtime.InteropServices;

namespace Sts2Sim.GodotShim;

/// <summary>ABI mirror of 16-byte Godot handle structs (callable, signal, packed arrays): two INTEGER eightbytes.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct R16 { public nint A, B; }

/// <summary>ABI mirror of godot_variant (24 bytes, returned via hidden pointer).</summary>
[StructLayout(LayoutKind.Sequential)]
public struct R24 { public nint A, B, C; }

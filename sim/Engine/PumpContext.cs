using System.Collections.Concurrent;

namespace Sts2Sim.Engine;

/// <summary>
/// Single-threaded SynchronizationContext. Every await inside the game logic resumes
/// through this queue on the simulator thread, so execution is deterministic and only
/// advances when <see cref="Drain"/> is called.
/// </summary>
public sealed class PumpContext : SynchronizationContext
{
    readonly ConcurrentQueue<(SendOrPostCallback cb, object? state)> _queue = new();
    readonly SemaphoreSlim _signal = new(0);
    readonly int _threadId = Environment.CurrentManagedThreadId;

    readonly List<TaskCompletionSource> _ticks = new();

    public static PumpContext? Instance { get; private set; }

    public static PumpContext Install()
    {
        var ctx = new PumpContext();
        SetSynchronizationContext(ctx);
        Instance = ctx;
        return ctx;
    }

    /// <summary>
    /// Replacement for real-time waits in game logic (Task.Delay): completes when the driver next runs
    /// out of queued work (<see cref="ReleaseTicks"/>), so polling loops still yield but no time passes.
    /// </summary>
    public static Task Delay(int msec)
    {
        var ctx = Instance;
        if (ctx == null || Environment.CurrentManagedThreadId != ctx._threadId) return Task.Delay(msec);
        var tcs = new TaskCompletionSource();
        ctx._ticks.Add(tcs);
        return tcs.Task;
    }

    /// <summary>Complete all pending <see cref="Delay"/> tasks. Returns false if there were none.</summary>
    public bool ReleaseTicks()
    {
        if (_ticks.Count == 0) return false;
        var ticks = _ticks.ToArray();
        _ticks.Clear();
        foreach (var t in ticks) t.SetResult();
        return true;
    }

    public override void Post(SendOrPostCallback d, object? state)
    {
        _queue.Enqueue((d, state));
        _signal.Release();
    }

    public override void Send(SendOrPostCallback d, object? state)
    {
        if (Environment.CurrentManagedThreadId == _threadId) d(state);
        else
        {
            using var done = new ManualResetEventSlim();
            Post(_ => { d(state); done.Set(); }, null);
            done.Wait();
        }
    }

    public override SynchronizationContext CreateCopy() => this;

    public bool HasWork => !_queue.IsEmpty;

    /// <summary>Run queued continuations until the queue is empty. Returns number executed.</summary>
    public int Drain(int maxSteps = 10_000_000)
    {
        int n = 0;
        while (n < maxSteps && _queue.TryDequeue(out var item))
        {
            _signal.Wait(0);
            item.cb(item.state);
            n++;
        }
        if (n >= maxSteps) throw new InvalidOperationException("PumpContext: runaway continuation loop");
        return n;
    }

    /// <summary>Block until work is posted (e.g. a Task.Delay timer fires) or the timeout elapses.</summary>
    public bool WaitForWork(TimeSpan timeout) => !_queue.IsEmpty || _signal.Wait(timeout);
}

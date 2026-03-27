// SafeHandle subclasses for automatic native resource cleanup.

using System.Runtime.InteropServices;

namespace Obrain.Native;

/// <summary>
/// Safe handle wrapping a <c>ObrainDatabase*</c>.
/// Calls <c>obrain_close</c> + <c>obrain_free_database</c> on release.
/// </summary>
internal sealed class DatabaseHandle : SafeHandle
{
    public DatabaseHandle() : base(nint.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == nint.Zero;

    protected override bool ReleaseHandle()
    {
        NativeMethods.obrain_close(handle);
        NativeMethods.obrain_free_database(handle);
        return true;
    }
}

/// <summary>
/// Safe handle wrapping a <c>ObrainTransaction*</c>.
/// Auto-rolls back and frees on release.
/// </summary>
internal sealed class TransactionHandle : SafeHandle
{
    public TransactionHandle() : base(nint.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == nint.Zero;

    protected override bool ReleaseHandle()
    {
        // The C side auto-rolls back if not committed, but be explicit.
        NativeMethods.obrain_rollback(handle);
        NativeMethods.obrain_free_transaction(handle);
        return true;
    }
}

/// <summary>
/// Safe handle wrapping a <c>ObrainResult*</c>.
/// Frees the result on release.
/// </summary>
internal sealed class ResultHandle : SafeHandle
{
    public ResultHandle() : base(nint.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == nint.Zero;

    protected override bool ReleaseHandle()
    {
        NativeMethods.obrain_free_result(handle);
        return true;
    }
}

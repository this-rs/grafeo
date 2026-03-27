// Error handling: status codes, exception hierarchy, and error retrieval.

using System.Runtime.InteropServices;

using Obrain.Native;

namespace Obrain;

/// <summary>
/// Status codes returned by obrain-c FFI functions.
/// Values match the <c>ObrainStatus</c> repr(C) enum in <c>error.rs</c>.
/// </summary>
public enum ObrainStatus
{
    Ok = 0,
    Database = 1,
    Query = 2,
    Transaction = 3,
    Storage = 4,
    Io = 5,
    Serialization = 6,
    Internal = 7,
    NullPointer = 8,
    InvalidUtf8 = 9,
}

/// <summary>Base exception for all Obrain errors.</summary>
public class ObrainException : Exception
{
    /// <summary>The native status code that produced this error.</summary>
    public ObrainStatus Status { get; }

    public ObrainException(string message, ObrainStatus status)
        : base(message) => Status = status;

    public ObrainException(string message, ObrainStatus status, Exception innerException)
        : base(message, innerException) => Status = status;

    /// <summary>
    /// Retrieve the last error message from the native layer and return a
    /// typed exception.
    /// </summary>
    internal static ObrainException FromLastError(
        ObrainStatus fallbackStatus = ObrainStatus.Database)
    {
        var errorPtr = NativeMethods.obrain_last_error();
        var message = errorPtr != nint.Zero
            ? Marshal.PtrToStringUTF8(errorPtr) ?? "Unknown error"
            : "Unknown error";
        return Classify(fallbackStatus, message);
    }

    /// <summary>
    /// Map a native status code to the appropriate exception subclass.
    /// Mirrors <c>obrain_bindings_common::error::classify_error</c>.
    /// </summary>
    internal static ObrainException Classify(int statusCode, string message) =>
        Classify((ObrainStatus)statusCode, message);

    internal static ObrainException Classify(ObrainStatus status, string message) =>
        status switch
        {
            ObrainStatus.Query => new QueryException(message),
            ObrainStatus.Transaction => new TransactionException(message),
            ObrainStatus.Storage or ObrainStatus.Io => new StorageException(message),
            ObrainStatus.Serialization => new SerializationException(message),
            _ => new ObrainException(message, status),
        };

    /// <summary>
    /// Throw if <paramref name="status"/> is not <see cref="ObrainStatus.Ok"/>.
    /// </summary>
    internal static void ThrowIfFailed(int status)
    {
        if (status != (int)ObrainStatus.Ok)
            throw FromLastError((ObrainStatus)status);
    }
}

/// <summary>Query parsing or execution error.</summary>
public sealed class QueryException(string message)
    : ObrainException(message, ObrainStatus.Query);

/// <summary>Transaction lifecycle error (commit, rollback, isolation).</summary>
public sealed class TransactionException(string message)
    : ObrainException(message, ObrainStatus.Transaction);

/// <summary>Storage or I/O error (WAL, persistence, disk).</summary>
public sealed class StorageException(string message)
    : ObrainException(message, ObrainStatus.Storage);

/// <summary>JSON or value serialization error.</summary>
public sealed class SerializationException(string message)
    : ObrainException(message, ObrainStatus.Serialization);

using Xunit;

namespace Obrain.Tests;

/// <summary>Database lifecycle, info, and admin tests.</summary>
public sealed class DatabaseTests : IDisposable
{
    private readonly ObrainDB _db = ObrainDB.Memory();

    public void Dispose() => _db.Dispose();

    [Fact]
    public void OpensInMemoryDatabase()
    {
        Assert.Equal(0, _db.NodeCount);
        Assert.Equal(0, _db.EdgeCount);
    }

    [Fact]
    public void DoubleDisposeIsNoOp()
    {
        using var db = ObrainDB.Memory();
        db.Dispose();
        db.Dispose(); // should not throw
    }

    [Fact]
    public void ThrowsOnUseAfterDispose()
    {
        var db = ObrainDB.Memory();
        db.Dispose();
        Assert.Throws<ObjectDisposedException>(() => db.Execute("RETURN 1"));
    }

    [Fact]
    public void ReturnsVersion()
    {
        var version = ObrainDB.Version;
        Assert.NotNull(version);
        Assert.NotEqual("unknown", version);
        Assert.Contains('.', version); // semver: X.Y.Z
    }

    [Fact]
    public void ReturnsInfo()
    {
        var info = _db.Info();
        Assert.NotNull(info);
        Assert.True(info.Count > 0);
    }

    [Fact]
    public void NodeCountReflectsInserts()
    {
        Assert.Equal(0, _db.NodeCount);
        _db.Execute("INSERT (:Person {name: 'Alix'})");
        Assert.Equal(1, _db.NodeCount);
        _db.Execute("INSERT (:Person {name: 'Gus'})");
        Assert.Equal(2, _db.NodeCount);
    }

    [Fact]
    public void EdgeCountReflectsInserts()
    {
        _db.Execute("INSERT (:Person {name: 'Vincent'})-[:KNOWS]->(:Person {name: 'Jules'})");
        Assert.Equal(2, _db.NodeCount);
        Assert.Equal(1, _db.EdgeCount);
    }
}

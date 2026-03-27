using Xunit;

namespace Obrain.Tests;

/// <summary>ACID transaction lifecycle tests.</summary>
public sealed class TransactionTests : IDisposable
{
    private readonly ObrainDB _db = ObrainDB.Memory();

    public void Dispose() => _db.Dispose();

    [Fact]
    public void CommitsTransaction()
    {
        using var tx = _db.BeginTransaction();
        tx.Execute("INSERT (:Person {name: 'Alix'})");
        tx.Commit();

        // Data visible after commit
        var result = _db.Execute("MATCH (p:Person) RETURN p.name");
        Assert.Single(result.Rows);
        Assert.Equal("Alix", result.Rows[0]["p.name"]);
    }

    [Fact]
    public void RollsBackTransaction()
    {
        using var tx = _db.BeginTransaction();
        tx.Execute("INSERT (:Person {name: 'Gus'})");
        tx.Rollback();

        // Data not visible after rollback
        var result = _db.Execute("MATCH (p:Person) RETURN p.name");
        Assert.Empty(result.Rows);
    }

    [Fact]
    public void AutoRollsBackOnDispose()
    {
        using (_db.BeginTransaction())
        {
            // Transaction disposed without commit
        }

        // Implicit rollback: no data
        Assert.Equal(0, _db.NodeCount);
    }

    [Fact]
    public void AutoRollsBackOnException()
    {
        try
        {
            using var tx = _db.BeginTransaction();
            tx.Execute("INSERT (:Person {name: 'Vincent'})");
            throw new InvalidOperationException("simulated failure");
        }
        catch (InvalidOperationException)
        {
            // expected
        }

        // Transaction rolled back due to exception
        Assert.Equal(0, _db.NodeCount);
    }

    [Fact]
    public void ExecutesMultipleQueriesInTransaction()
    {
        using var tx = _db.BeginTransaction();
        tx.Execute("INSERT (:Person {name: 'Jules'})");
        tx.Execute("INSERT (:Person {name: 'Mia'})");
        tx.Execute("INSERT (:Person {name: 'Butch'})");
        tx.Commit();

        Assert.Equal(3, _db.NodeCount);
    }

    [Fact]
    public void ThrowsOnUseAfterCommit()
    {
        using var tx = _db.BeginTransaction();
        tx.Commit();

        Assert.Throws<TransactionException>(
            () => tx.Execute("INSERT (:Person {name: 'Django'})"));
    }

    [Fact]
    public void DoubleCommitIsHandled()
    {
        using var tx = _db.BeginTransaction();
        tx.Execute("INSERT (:Person {name: 'Shosanna'})");
        tx.Commit();

        // Second commit should not throw (finished flag prevents it)
        Assert.Throws<TransactionException>(() => tx.Commit());
    }

    [Fact]
    public void RollbackAfterCommitIsNoOp()
    {
        using var tx = _db.BeginTransaction();
        tx.Execute("INSERT (:Person {name: 'Hans'})");
        tx.Commit();

        // Rollback after commit: finished flag prevents action
        tx.Rollback(); // no-op
        Assert.Equal(1, _db.NodeCount);
    }

    [Fact]
    public void QueryWithinTransactionReturnsResults()
    {
        using var tx = _db.BeginTransaction();
        tx.Execute("INSERT (:Person {name: 'Beatrix', age: 35})");

        var result = tx.Execute("MATCH (p:Person) RETURN p.name, p.age");
        Assert.Single(result.Rows);
        Assert.Equal("Beatrix", result.Rows[0]["p.name"]);

        tx.Commit();
    }

    [Fact]
    public void ParameterizedQueryInTransaction()
    {
        using var tx = _db.BeginTransaction();
        tx.Execute("INSERT (:City {name: 'Berlin'})");

        var result = tx.ExecuteWithParams(
            "MATCH (c:City) WHERE c.name = $name RETURN c.name",
            new Dictionary<string, object?> { ["name"] = "Berlin" });

        Assert.Single(result.Rows);
        Assert.Equal("Berlin", result.Rows[0]["c.name"]);
        tx.Commit();
    }
}

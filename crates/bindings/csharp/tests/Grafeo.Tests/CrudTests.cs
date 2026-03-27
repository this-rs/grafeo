using Xunit;

namespace Obrain.Tests;

/// <summary>Node and edge CRUD operation tests.</summary>
public sealed class CrudTests : IDisposable
{
    private readonly ObrainDB _db = ObrainDB.Memory();

    public void Dispose() => _db.Dispose();

    // -- Node CRUD --

    [Fact]
    public void CreatesAndGetsNode()
    {
        var id = _db.CreateNode(["Person"], new Dictionary<string, object?>
        {
            ["name"] = "Alix",
            ["age"] = 30,
        });

        Assert.True(id >= 0);
        var node = _db.GetNode(id);
        Assert.NotNull(node);
        Assert.Equal(id, node.Id);
        Assert.Contains("Person", node.Labels);
        Assert.Equal("Alix", node.Properties["name"]);
        Assert.Equal(30L, node.Properties["age"]);
    }

    [Fact]
    public void DeletesNode()
    {
        var id = _db.CreateNode(["Person"]);
        Assert.Equal(1, _db.NodeCount);

        var deleted = _db.DeleteNode(id);
        Assert.True(deleted);
        Assert.Equal(0, _db.NodeCount);
    }

    [Fact]
    public void DeleteNonExistentNodeReturnsFalse()
    {
        Assert.False(_db.DeleteNode(999999));
    }

    [Fact]
    public void GetNonExistentNodeReturnsNull()
    {
        Assert.Null(_db.GetNode(999999));
    }

    [Fact]
    public void SetsNodeProperty()
    {
        var id = _db.CreateNode(["Person"], new Dictionary<string, object?>
        {
            ["name"] = "Gus"
        });

        _db.SetNodeProperty(id, "age", 25);

        var node = _db.GetNode(id);
        Assert.NotNull(node);
        Assert.Equal(25L, node.Properties["age"]);
    }

    [Fact]
    public void RemovesNodeProperty()
    {
        var id = _db.CreateNode(["Person"], new Dictionary<string, object?>
        {
            ["name"] = "Vincent",
            ["age"] = 40,
        });

        var removed = _db.RemoveNodeProperty(id, "age");
        Assert.True(removed);

        var node = _db.GetNode(id);
        Assert.NotNull(node);
        Assert.False(node.Properties.ContainsKey("age"));
    }

    [Fact]
    public void AddsNodeLabel()
    {
        var id = _db.CreateNode(["Person"]);
        var added = _db.AddNodeLabel(id, "Employee");
        Assert.True(added);

        var node = _db.GetNode(id);
        Assert.NotNull(node);
        Assert.Contains("Person", node.Labels);
        Assert.Contains("Employee", node.Labels);
    }

    [Fact]
    public void RemovesNodeLabel()
    {
        var id = _db.CreateNode(["Person", "Employee"]);
        var removed = _db.RemoveNodeLabel(id, "Employee");
        Assert.True(removed);

        var node = _db.GetNode(id);
        Assert.NotNull(node);
        Assert.Contains("Person", node.Labels);
        Assert.DoesNotContain("Employee", node.Labels);
    }

    // -- Edge CRUD --

    [Fact]
    public void CreatesAndGetsEdge()
    {
        var alix = _db.CreateNode(["Person"], new Dictionary<string, object?> { ["name"] = "Alix" });
        var gus = _db.CreateNode(["Person"], new Dictionary<string, object?> { ["name"] = "Gus" });

        var edgeId = _db.CreateEdge(alix, gus, "KNOWS", new Dictionary<string, object?>
        {
            ["since"] = 2020,
        });

        Assert.True(edgeId >= 0);
        var edge = _db.GetEdge(edgeId);
        Assert.NotNull(edge);
        Assert.Equal(edgeId, edge.Id);
        Assert.Equal("KNOWS", edge.Type);
        Assert.Equal(alix, edge.SourceId);
        Assert.Equal(gus, edge.TargetId);
        Assert.Equal(2020L, edge.Properties["since"]);
    }

    [Fact]
    public void DeletesEdge()
    {
        var a = _db.CreateNode(["Person"]);
        var b = _db.CreateNode(["Person"]);
        var eid = _db.CreateEdge(a, b, "KNOWS");
        Assert.Equal(1, _db.EdgeCount);

        Assert.True(_db.DeleteEdge(eid));
        Assert.Equal(0, _db.EdgeCount);
    }

    [Fact]
    public void GetNonExistentEdgeReturnsNull()
    {
        Assert.Null(_db.GetEdge(999999));
    }

    [Fact]
    public void SetsEdgeProperty()
    {
        var a = _db.CreateNode(["Person"]);
        var b = _db.CreateNode(["Person"]);
        var eid = _db.CreateEdge(a, b, "KNOWS");

        _db.SetEdgeProperty(eid, "weight", 0.9);

        var edge = _db.GetEdge(eid);
        Assert.NotNull(edge);
        Assert.Equal(0.9, edge.Properties["weight"]);
    }

    [Fact]
    public void RemovesEdgeProperty()
    {
        var a = _db.CreateNode(["Person"]);
        var b = _db.CreateNode(["Person"]);
        var eid = _db.CreateEdge(a, b, "KNOWS", new Dictionary<string, object?>
        {
            ["since"] = 2020,
        });

        var removed = _db.RemoveEdgeProperty(eid, "since");
        Assert.True(removed);

        var edge = _db.GetEdge(eid);
        Assert.NotNull(edge);
        Assert.False(edge.Properties.ContainsKey("since"));
    }
}

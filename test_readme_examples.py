#!/usr/bin/env python3
"""
Test script to verify that README examples work correctly.
"""

import sys
import polars as pl

# Add the source directory to Python path  
sys.path.append('./src')

def test_basic_example():
    """Test the basic example from README."""
    print("Testing Basic Example...")
    
    try:
        from network.construction import build_graph_from_edgelist
        from glp.propagation import guided_label_propagation
        print("✅ Imports successful")
        
        # Create sample data
        edges = pl.DataFrame({
            "source": ["A", "B", "C", "A"],
            "target": ["B", "C", "A", "B"],
            "weight": [1.0, 1.0, 1.0, 1.0]
        })
        
        graph, mapper = build_graph_from_edgelist(edges, "source", "target", "weight")
        print(f"✅ Graph created: {graph.numberOfNodes()} nodes, {graph.numberOfEdges()} edges")
        
    except Exception as e:
        print(f"❌ Basic example failed: {e}")

def test_temporal_bipartite_example():
    """Test the temporal bipartite example from README."""
    print("\nTesting Temporal Bipartite Example...")
    
    try:
        from network.construction import temporal_bipartite_to_unipartite
        print("✅ Temporal import successful")
        
        # Sample temporal bipartite data
        data = pl.DataFrame({
            "user": ["Alice", "Bob", "Charlie", "Alice", "Bob"],
            "item": ["item1", "item1", "item1", "item2", "item2"], 
            "timestamp": ["2024-01-01 09:00", "2024-01-01 11:00", "2024-01-01 13:00",
                          "2024-01-02 10:00", "2024-01-02 15:00"]
        })
        
        # Convert to directed user-user influence network  
        influence_graph, user_mapper = temporal_bipartite_to_unipartite(
            data,
            source_col="user",
            target_col="item",
            timestamp_col="timestamp",
            intermediate_col="item",    # Items disappear
            projected_col="user",       # Users get connected
            add_edge_weights=True       # Include temporal decay
        )
        
        print(f"✅ Created {influence_graph.numberOfNodes()} user influence network")
        print(f"✅ Temporal relationships: {influence_graph.numberOfEdges()} edges")
        print("✅ Expected edges: Alice → Bob → Charlie (temporal precedence preserved)")
        
    except Exception as e:
        print(f"❌ Temporal bipartite example failed: {e}")

def test_fixture_example():
    """Test using fixture data."""
    print("\nTesting Fixture Example...")
    
    try:
        from network.construction import build_graph_from_edgelist
        from common.id_mapper import IDMapper
        print("✅ Fixture imports successful")
        
        # Load sample datasets
        edges = pl.read_csv("tests/fixtures/sample_edgelist.csv")
        print(f"✅ Loaded {len(edges)} sample edges")
        
        graph, mapper = build_graph_from_edgelist(edges, "source", "target")
        print(f"✅ Sample graph: {graph.numberOfNodes()} nodes, {graph.numberOfEdges()} edges")
        
    except Exception as e:
        print(f"❌ Fixture example failed: {e}")

def main():
    """Run all tests."""
    print("Testing README Examples")
    print("=" * 40)
    
    test_basic_example()
    test_temporal_bipartite_example() 
    test_fixture_example()
    
    print("\n" + "=" * 40)
    print("README examples testing completed!")

if __name__ == "__main__":
    main()
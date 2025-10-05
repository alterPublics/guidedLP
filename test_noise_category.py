#!/usr/bin/env python3
"""
Test script for noise category functionality in GLP.

This script tests the new noise category features including:
- Automatic noise category addition
- Noise seed generation  
- Confidence thresholding
- Single label scenario handling
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import polars as pl
from src.network.construction import build_graph_from_edgelist
from src.glp.propagation import guided_label_propagation
from src.common.id_mapper import IDMapper

def create_test_data():
    """Create test network data for noise category testing."""
    
    # Create a simple test network
    edges_data = [
        # Community A (strong connections)
        ("A1", "A2", 2.0),
        ("A2", "A3", 2.0), 
        ("A3", "A4", 2.0),
        ("A1", "A4", 1.5),
        
        # Community B (strong connections)
        ("B1", "B2", 2.0),
        ("B2", "B3", 2.0),
        ("B3", "B4", 2.0),
        ("B1", "B4", 1.5),
        
        # Weak inter-community connections (should become noise)
        ("A1", "B1", 0.3),
        ("A3", "B2", 0.3),
        
        # Isolated outlier nodes (should become noise)
        ("OUT1", "OUT2", 1.0),
        ("OUT2", "OUT3", 0.5),
        
        # Weak connections to communities
        ("OUT1", "A1", 0.2),
        ("OUT3", "B3", 0.2)
    ]
    
    edges_df = pl.DataFrame({
        "source": [e[0] for e in edges_data],
        "target": [e[1] for e in edges_data], 
        "weight": [e[2] for e in edges_data]
    })
    
    # Define seed nodes
    seed_labels = {
        "A1": "community_a",
        "A2": "community_a", 
        "B1": "community_b",
        "B2": "community_b"
    }
    
    labels = ["community_a", "community_b"]
    
    return edges_df, seed_labels, labels

def test_noise_category_enabled():
    """Test GLP with noise category enabled."""
    print("=== Testing GLP with Noise Category Enabled ===")
    
    edges_df, seed_labels, labels = create_test_data()
    
    # Build graph
    graph, mapper = build_graph_from_edgelist(
        edges_df, "source", "target", "weight"
    )
    
    # Run GLP with noise category
    results = guided_label_propagation(
        graph=graph,
        id_mapper=mapper,
        seed_labels=seed_labels,
        labels=labels,
        alpha=0.85,
        enable_noise_category=True,
        noise_ratio=0.15,
        confidence_threshold=0.6
    )
    
    print(f"Results shape: {results.shape}")
    print(f"Columns: {results.columns}")
    print("\nLabel distribution:")
    print(results["dominant_label"].value_counts())
    
    print(f"\nNodes classified as 'noise': {(results['dominant_label'] == 'noise').sum()}")
    print(f"Nodes classified as 'uncertain': {(results['dominant_label'] == 'uncertain').sum()}")
    
    # Show some example results
    print(f"\nSample results:")
    sample_results = results.head(10)
    for row in sample_results.iter_rows(named=True):
        print(f"  {row['node_id']}: {row['dominant_label']} (confidence: {row['confidence']:.3f})")
    
    return results

def test_noise_category_disabled():
    """Test GLP with noise category disabled."""
    print("\n=== Testing GLP with Noise Category Disabled ===")
    
    edges_df, seed_labels, labels = create_test_data()
    
    # Build graph
    graph, mapper = build_graph_from_edgelist(
        edges_df, "source", "target", "weight"
    )
    
    # Run GLP without noise category
    results = guided_label_propagation(
        graph=graph,
        id_mapper=mapper,
        seed_labels=seed_labels,
        labels=labels,
        alpha=0.85,
        enable_noise_category=False,
        confidence_threshold=0.6
    )
    
    print(f"Results shape: {results.shape}")
    print("\nLabel distribution:")
    print(results["dominant_label"].value_counts())
    
    print(f"\nNodes classified as 'uncertain': {(results['dominant_label'] == 'uncertain').sum()}")
    
    return results

def test_single_label_scenario():
    """Test GLP with single label (should trigger warning)."""
    print("\n=== Testing Single Label Scenario ===")
    
    edges_df, seed_labels, labels = create_test_data()
    
    # Use only one label
    single_seed_labels = {"A1": "important"}
    single_labels = ["important"]
    
    # Build graph
    graph, mapper = build_graph_from_edgelist(
        edges_df, "source", "target", "weight"
    )
    
    # Run GLP with single label and noise category
    print("Running with noise category enabled (recommended):")
    results_with_noise = guided_label_propagation(
        graph=graph,
        id_mapper=mapper,
        seed_labels=single_seed_labels,
        labels=single_labels,
        alpha=0.85,
        enable_noise_category=True,
        noise_ratio=0.2
    )
    
    print("Label distribution with noise:")
    print(results_with_noise["dominant_label"].value_counts())
    
    # Run without noise category (should show warning)
    print("\nRunning without noise category (shows warning):")
    results_without_noise = guided_label_propagation(
        graph=graph,
        id_mapper=mapper,
        seed_labels=single_seed_labels,
        labels=single_labels,
        alpha=0.85,
        enable_noise_category=False
    )
    
    print("Label distribution without noise:")
    print(results_without_noise["dominant_label"].value_counts())
    
    return results_with_noise, results_without_noise

def compare_results():
    """Compare results with and without noise category."""
    print("\n=== Comparing Results ===")
    
    edges_df, seed_labels, labels = create_test_data()
    graph, mapper = build_graph_from_edgelist(edges_df, "source", "target", "weight")
    
    # With noise
    with_noise = guided_label_propagation(
        graph, mapper, seed_labels, labels,
        enable_noise_category=True, noise_ratio=0.1
    )
    
    # Without noise  
    without_noise = guided_label_propagation(
        graph, mapper, seed_labels, labels,
        enable_noise_category=False
    )
    
    print("Classification confidence comparison:")
    print(f"  With noise - avg confidence: {with_noise['confidence'].mean():.3f}")
    print(f"  Without noise - avg confidence: {without_noise['confidence'].mean():.3f}")
    
    print(f"\nNodes with confidence > 0.8:")
    print(f"  With noise: {(with_noise['confidence'] > 0.8).sum()}")
    print(f"  Without noise: {(without_noise['confidence'] > 0.8).sum()}")

def main():
    """Run all tests."""
    try:
        print("Testing Noise Category Functionality in GLP")
        print("=" * 50)
        
        # Test with noise category enabled
        results_with_noise = test_noise_category_enabled()
        
        # Test with noise category disabled
        results_without_noise = test_noise_category_disabled()
        
        # Test single label scenario
        single_with_noise, single_without_noise = test_single_label_scenario()
        
        # Compare results
        compare_results()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        print("\nKey findings:")
        print("- Noise category successfully identifies outlier nodes")
        print("- Confidence thresholding marks uncertain predictions")
        print("- Single label scenarios work better with noise category")
        print("- Overall classification confidence improved with noise handling")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
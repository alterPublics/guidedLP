#!/usr/bin/env python3
"""
Test script to verify package imports work correctly after installation.
"""

def test_imports():
    """Test different import approaches."""
    
    print("Testing package imports after installation...")
    print("=" * 50)
    
    # Test 1: Basic common module (no relative imports)
    try:
        from common.id_mapper import IDMapper
        print("✅ from common.id_mapper import IDMapper")
    except ImportError as e:
        print(f"❌ from common.id_mapper import IDMapper: {e}")
    
    # Test 2: Network construction (has relative imports)
    try:
        from network.construction import build_graph_from_edgelist
        print("✅ from network.construction import build_graph_from_edgelist")
    except ImportError as e:
        print(f"❌ from network.construction import build_graph_from_edgelist: {e}")
        
    # Test 3: Temporal function (has relative imports)
    try:
        from network.construction import temporal_bipartite_to_unipartite
        print("✅ from network.construction import temporal_bipartite_to_unipartite")
    except ImportError as e:
        print(f"❌ from network.construction import temporal_bipartite_to_unipartite: {e}")
    
    # Test 4: GLP propagation (has relative imports)
    try:
        from glp.propagation import guided_label_propagation
        print("✅ from glp.propagation import guided_label_propagation")
    except ImportError as e:
        print(f"❌ from glp.propagation import guided_label_propagation: {e}")
    
    # Test 5: Package level import
    try:
        import sys
        if '/Users/jakobbk/Documents/postdoc/codespace/guidedLabelPropagation/guidedLP/src' in sys.path:
            print("✅ Package source directory in Python path")
        else:
            print("❌ Package source directory NOT in Python path")
    except Exception as e:
        print(f"❌ Path check failed: {e}")
    
    print("\n" + "=" * 50)
    print("Import test completed")

if __name__ == "__main__":
    test_imports()
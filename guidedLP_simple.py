#!/usr/bin/env python3
"""
Simple convenience module for guidedLP package.

This module provides easy imports without dealing with relative import issues.
Use this for simple scripts and examples.

Usage:
    # Add to your Python path
    import sys
    sys.path.append('/path/to/guidedLabelPropagation/guidedLP')
    
    from guidedLP_simple import guided_label_propagation, build_graph_from_edgelist
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
_src_path = Path(__file__).parent / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# Import working modules (ones without problematic relative imports)
try:
    from common.id_mapper import IDMapper
    from common.exceptions import (
        GraphConstructionError, 
        ValidationError, 
        DataFormatError,
        ConfigurationError
    )
    __all__ = ["IDMapper", "GraphConstructionError", "ValidationError", "DataFormatError", "ConfigurationError"]
    print("✅ Successfully imported common modules")
except ImportError as e:
    print(f"❌ Failed to import common modules: {e}")
    __all__ = []

# Note about modules with relative imports
_note = """
NOTE: Some modules have relative import issues when used as a package.
For these modules, use the original approach:

import sys
sys.path.append('/path/to/guidedLabelPropagation/guidedLP/src')

# Then use direct imports:
from network.construction import build_graph_from_edgelist, temporal_bipartite_to_unipartite
from glp.propagation import guided_label_propagation

These will work when the src directory is in your Python path.
"""

def print_usage_info():
    """Print usage information for the package."""
    print("GuidedLP Package Usage Information")
    print("=" * 40)
    print()
    print("Current working approach:")
    print("1. Add src directory to Python path:")
    print("   import sys")
    print("   sys.path.append('/path/to/guidedLabelPropagation/guidedLP/src')")
    print()
    print("2. Import modules directly:")
    print("   from network.construction import build_graph_from_edgelist, temporal_bipartite_to_unipartite")
    print("   from glp.propagation import guided_label_propagation")
    print("   from common.id_mapper import IDMapper")
    print()
    print("3. Use the functions normally:")
    print("   graph, mapper = build_graph_from_edgelist(edges)")
    print("   results = guided_label_propagation(graph, mapper, seeds, labels)")
    print()
    print("Available modules:")
    print("- network.construction: Graph building and temporal bipartite conversion")
    print("- glp.propagation: Guided label propagation algorithm")
    print("- network.analysis: Network centrality and analysis")
    print("- timeseries.slicing: Temporal network slicing")
    print("- common.id_mapper: ID mapping utilities")

if __name__ == "__main__":
    print_usage_info()
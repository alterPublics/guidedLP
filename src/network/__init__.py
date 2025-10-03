"""
Network construction and analysis module.

This module provides core network analysis capabilities:
- Graph construction from edge lists (unipartite and bipartite)
- Bipartite graph projection
- Network backboning and filtering
- Centrality measure calculations
- Community detection using Louvain algorithm
- Graph export functionality
"""

# Network construction functions
from .construction import (
    build_graph_from_edgelist,
    project_bipartite,
    get_graph_info,
    get_bipartite_info,
    validate_graph_construction
)

# Imports will be added as modules are implemented
# from .analysis import CentralityCalculator, CommunityDetector
# from .backboning import NetworkBackbone
"""
Network construction and analysis module.

This module provides core network analysis capabilities:
- Graph construction from edge lists (unipartite and bipartite)
- Bipartite graph projection
- Centrality measure calculations (degree, betweenness, closeness, eigenvector, pagerank, katz)
- Community detection using Louvain algorithm with consensus and stability analysis
- Network backboning and filtering (planned)
- Graph export functionality (planned)
"""

# Network construction functions
from .construction import (
    build_graph_from_edgelist,
    project_bipartite,
    get_graph_info,
    get_bipartite_info,
    validate_graph_construction
)

# Network analysis functions
from .analysis import (
    extract_centrality,
    get_centrality_summary,
    identify_central_nodes
)

# Community detection functions
from .communities import (
    detect_communities,
    get_community_summary,
    identify_stable_communities
)

# Imports will be added as modules are implemented
# from .backboning import apply_backbone
# from .filtering import filter_graph
# from .export import export_graph
# Guided Label Propagation (GLP)

Large-scale network analysis with semi-supervised community detection for computational social science research.

## Overview

This project provides efficient network analysis capabilities with a focus on **Guided Label Propagation (GLP)**, a novel semi-supervised community detection technique. Unlike traditional unsupervised methods that find arbitrary clusters, GLP identifies how unknown nodes in a network relate to predefined categories of interest (e.g., political affiliation, brand preference, topic relevance).

## Key Features

### 🚀 High-Performance Network Analysis
- **Large-scale optimization**: Designed for networks with 10,000+ nodes
- **NetworkIt backend**: Leverages C++ performance for graph operations  
- **Sparse matrix operations**: Memory-efficient computations using SciPy
- **Parallel processing**: Multi-threaded operations where beneficial

### 🎯 Guided Label Propagation (GLP)
- **Semi-supervised approach**: Uses seed nodes to guide community detection
- **Directional propagation**: Supports both in-degree and out-degree based propagation
- **Probability estimation**: Calculates affinity scores for unknown nodes
- **Validation framework**: Built-in train/test split and external validation

### 📊 Comprehensive Network Toolkit
- **Graph construction**: Unipartite and bipartite networks from edge lists
- **Network backboning**: Statistical significance filtering
- **Centrality measures**: Degree, betweenness, closeness, eigenvector centrality
- **Community detection**: Louvain algorithm integration
- **Temporal analysis**: Time-sliced network evolution

### 🔄 Flexible Data Pipeline
- **Polars integration**: Fast DataFrame operations for large datasets
- **Multiple formats**: Support for CSV, Parquet input/output
- **ID preservation**: Maintains original node identifiers throughout analysis
- **Export options**: GEXF, GraphML, CSV outputs

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/guided-label-propagation.git
cd guided-label-propagation

# Install with pip
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
import polars as pl
from guided_lp import NetworkBuilder, GuidedLabelPropagation

# Load edge list data
edges = pl.read_csv("network_data.csv")

# Build network
builder = NetworkBuilder()
graph, id_mapper = builder.from_edge_list(
    edges, 
    source_col="user_a", 
    target_col="user_b"
)

# Run Guided Label Propagation
glp = GuidedLabelPropagation(graph, id_mapper)
seed_nodes = {"progressive": ["user123", "user456"], "conservative": ["user789"]}
probabilities = glp.propagate(seed_nodes, max_iterations=100)

# Export results
glp.export_probabilities(probabilities, "affiliation_scores.csv")
```

## Architecture

The system is organized into three main modules:

```
src/
├── common/          # Shared utilities (ID mapping, validation, export)
├── network/         # Graph construction and analysis
├── glp/            # Guided Label Propagation implementation  
└── timeseries/     # Temporal network analysis
```

### Module Independence
- **Network module**: Standalone graph analysis capabilities
- **GLP module**: Requires network module, adds semi-supervised detection
- **Time-series module**: Temporal analysis, can work with or without GLP

## Performance Characteristics

- **Graph construction**: O(E + V) using NetworkIt
- **Label propagation**: O(I × E) where I is iterations, E is edges
- **Memory usage**: Sparse matrices for networks with >50% zero entries
- **Parallel support**: Multi-threaded centrality calculations and time-slicing

## Use Cases

### Political Affiliation Analysis
Identify political leaning of unknown users based on known partisan seed accounts.

### Brand Affinity Detection  
Determine brand preferences in social networks using verified brand accounts as seeds.

### Research Community Mapping
Map academic collaboration networks and identify research area affiliations.

### Temporal Network Evolution
Track how community structures evolve over time in dynamic networks.

## Documentation

- [Architecture Overview](docs/architecture/overview.md)
- [API Reference](docs/api/)
- [Performance Guidelines](docs/performance.md)
- [Examples](examples/)

## Requirements

- Python 3.9+
- NetworkIt 11.0+
- Polars 0.20.0+
- NumPy 1.24.0+
- SciPy 1.10.0+

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the code style (ruff + black)
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{guided_label_propagation,
  title={Guided Label Propagation: Semi-supervised Community Detection for Large-Scale Networks},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/guided-label-propagation}
}
```

## Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/guided-label-propagation/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/guided-label-propagation/discussions)
- **Email**: your.email@example.com
# Guided Label Propagation (GLP)

Large-scale network analysis with semi-supervised community detection for computational social science research.

## Overview

This project provides efficient network analysis capabilities with a focus on **Guided Label Propagation (GLP)**, a novel semi-supervised community detection technique. Unlike traditional unsupervised methods that find arbitrary clusters, GLP identifies how unknown nodes in a network relate to predefined categories of interest (e.g., political affiliation, brand preference, topic relevance).


## Installation

### Prerequisites

- Python 3.9 or higher
- Git (for development installation)

### Installation

Install the package directly with pip:

```bash
# Clone the repository
git clone https://github.com/alterpublics/guided-label-propagation.git
cd guided-label-propagation/guidedLP

# Install the package in development mode
pip install -e .

# Or install dependencies separately if needed
pip install -r requirements.txt
```

### Development Setup

For development work:

```bash
# Install development dependencies  
pip install pytest pytest-cov ruff black mypy
```

### Verify Installation

After installation, you can verify everything works correctly:

```bash
python test_installation.py
```

This will test all key functionality and confirm your installation is working properly.

# Or install with all optional dependencies
pip install -e ".[dev,docs,viz]"
```

### Optional Dependencies

```bash
# For visualization capabilities
pip install "guided-label-propagation[viz]"

# For development and testing
pip install "guided-label-propagation[dev]"

# For documentation building
pip install "guided-label-propagation[docs]"
```

### Verify Installation

```bash
python -c "import guided_lp; print('Installation successful!')"
```

## Quick Start

### Basic Example

```python
import polars as pl
from guidedLP.network.construction import build_graph_from_edgelist
from guidedLP.glp.propagation import guided_label_propagation

# Load edge list data
edges = pl.read_csv("network_data.csv")

# Build network
graph, id_mapper = build_graph_from_edgelist(
    edges, 
    source_col="user_a", 
    target_col="user_b",
    weight_col="weight"  # optional
)

# Define seed nodes for each community
seed_nodes = {
    "progressive": ["user123", "user456", "user789"],
    "conservative": ["user321", "user654", "user987"]
}

# Run Guided Label Propagation
results = guided_label_propagation(
    graph=graph,
    seeds=seed_nodes,
    id_mapper=id_mapper,
    max_iterations=100,
    threshold=0.01
)

# Export results
export_results(results, "political_affiliation_scores.csv")
print(f"Classified {len(results)} nodes with community probabilities")
```

# Build graph and run GLP
graph, id_mapper = build_graph_from_edgelist(edges, "source", "target", "weight")
results = guided_label_propagation(graph, seed_nodes, id_mapper)

print(f"Sample analysis complete: {len(results)} nodes classified")
```



## Examples

### 1. Political Affiliation Analysis

```python
# Analyze political leaning in social networks
from guidedLP.glp.validation import train_test_split_validation
from guidedLP.network.construction import build_graph_from_edgelist

# Load political Twitter network
political_edges = pl.read_csv("political_network.csv")
graph, id_mapper = build_graph_from_edgelist(
    political_edges, "follower", "following"
)

# Define known political accounts as seeds
political_seeds = {
    "progressive": ["@aoc", "@berniesanders", "@ewarren"],
    "conservative": ["@realdonaldtrump", "@tedcruz", "@marcorubio"]
}

# Run validation to test accuracy
accuracy, metrics = train_test_split_validation(
    graph=graph,
    seeds=political_seeds,
    id_mapper=id_mapper,
    test_size=0.2
)

print(f"Political classification accuracy: {accuracy:.3f}")
```

### 2. Temporal Network Analysis

```python
# Track community evolution over time
from guidedLP.timeseries.slicing import create_temporal_slices
from guidedLP.timeseries.temporal_metrics import extract_temporal_metrics

# Load temporal network data
temporal_data = pl.read_csv("tests/fixtures/sample_temporal.csv")

# Create time slices
time_slices = create_time_slices(
    temporal_data,
    time_col="timestamp",
    slice_duration="1d"  # daily slices
)

# Analyze each time slice
for date, slice_edges in time_slices.items():
    graph, id_mapper = build_graph_from_edgelist(
        slice_edges, "source", "target", "weight"
    )
    
    results = guided_label_propagation(graph, seeds, id_mapper)
    print(f"{date}: {len(results)} nodes classified")
```



## System Requirements

### Core Dependencies
- **Python**: 3.9 or higher
- **NetworkIt**: 11.0+ (C++ graph library for performance)
- **Polars**: 0.20.0+ (Fast DataFrame operations)
- **NumPy**: 1.24.0+ (Numerical computing)
- **SciPy**: 1.10.0+ (Sparse matrices and scientific computing)

### Performance Notes
- Minimum 8GB RAM recommended for networks with >10,000 nodes
- SSD storage recommended for large temporal datasets
- Multi-core CPU beneficial for parallel operations


## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

